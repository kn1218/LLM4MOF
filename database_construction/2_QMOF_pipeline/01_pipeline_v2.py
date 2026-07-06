"""QMOF v2 Enrichment Pipeline — deterministic pass.

Wires together all enrichment components to produce qmof_enriched_v2/ JSONs:
  1. Reads qmof/_source_data/qmof.csv (20,373 records)
  2. For each record: metal enrichment → SMARTS FG detection → validation → JSON
  3. Outputs one JSON per MOF in qmof/qmof_enriched_v2/{qmof_id}.json
  4. Logs all source issues from the P1 audit report into each record's issue_log
  5. LLM enrichment is an optional second pass (NOT in this pipeline)

Usage:
    python -m qmof.qmof_pipeline.pipeline_v2 --limit 5
    python -m qmof.qmof_pipeline.pipeline_v2 --skip-existing
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from qmof.qmof_pipeline.schema_v2 import (  # noqa: E402
    QMOFRecordV2,
    Layer1Facts,
    MetalNode,
    Layer2Semantics,
    LinkerEnrichment,
    FunctionalGroups,
    AbstractFeatures,
    Provenance,
    ValidationReport,
    IssueLogEntry,
    make_provenance,
)
from qmof.qmof_pipeline.metal_enricher import enrich_metal_node  # noqa: E402
from qmof.qmof_pipeline.smarts_engine import enrich_mof_smarts   # noqa: E402
from qmof.qmof_pipeline.validator_v2 import validate_record       # noqa: E402


# ── Path configuration ────────────────────────────────────────────────
SOURCE_CSV = BASE_DIR / "qmof" / "_source_data" / "qmof.csv"
SOURCE_JSON_DIR = BASE_DIR / "qmof" / "_source_data" / "qmof_global_jsons_v2"
AUDIT_REPORT = BASE_DIR / "qmof" / "_audit" / "source_investigation_report.json"
OUTPUT_DIR = BASE_DIR / "qmof" / "qmof_enriched_v2"


# ── SMARTS pattern count (lazy, from import chain) ───────────────────
try:
    from bb_pipeline.smarts_library import ALL_PATTERNS  # noqa: E402
    SMARTS_PATTERN_COUNT = len(ALL_PATTERNS)
except ImportError:
    SMARTS_PATTERN_COUNT = 0


# ── CSV column name mapping ──────────────────────────────────────────
# Actual CSV columns → schema_v2 Layer1Facts field names
COL = {
    "qmof_id":      "qmof_id",
    "formula":       "info.formula_reduced",
    "topology":      "info.mofid.topology",
    "smiles_nodes":  "info.mofid.smiles_nodes",
    "smiles_linkers":"info.mofid.smiles_linkers",
    "mofid":         "info.mofid.mofid",
    "mofkey":        "info.mofid.mofkey",
    "natoms":        "info.natoms",
    "density":       "info.density",
    "volume":        "info.volume",
    "pld":           "info.pld",
    "lcd":           "info.lcd",
    "spacegroup":    "info.symmetry.spacegroup",
    "crystal_system":"info.symmetry.spacegroup_crystal",
    "synthesized":   "info.synthesized",
    "doi":           "info.doi",
    "bandgap_pbe":   "outputs.pbe.bandgap",
    "bandgap_hle17": "outputs.hle17.bandgap",
    "bandgap_hse06_10hf": "outputs.hse06_10hf.bandgap",
    "bandgap_hse06": "outputs.hse06.bandgap",
}


# =====================================================================
# CSV value parsers
# =====================================================================

def _parse_smiles_list(raw: Any) -> list[str]:
    """Parse Python list string from CSV into actual list. Returns [] on failure."""
    if pd.isna(raw) if isinstance(raw, float) else (not raw):
        return []
    try:
        result = ast.literal_eval(str(raw))
        if isinstance(result, list):
            return [str(s) for s in result]
        return [str(result)]
    except Exception:
        return []


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float or return None."""
    try:
        f = float(val)
        return f if not pd.isna(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Convert to int or return None."""
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return int(f)
    except (ValueError, TypeError):
        return None


def _safe_str(val: Any) -> str:
    """Convert to string, return '' for NaN."""
    if isinstance(val, float) and pd.isna(val):
        return ""
    if pd.isna(val) if hasattr(val, '__class__') else False:
        return ""
    s = str(val)
    return "" if s == "nan" else s


def _safe_bool(val: Any) -> Optional[bool]:
    """Convert to bool or return None."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return None
        return bool(val)
    if isinstance(val, str):
        if val.lower() in ("true", "1", "yes"):
            return True
        if val.lower() in ("false", "0", "no"):
            return False
    return None


# =====================================================================
# SMILES validation (basic RDKit parse check for node SMILES)
# =====================================================================

def _validate_smiles_list(smiles_list: list[str]) -> tuple[list[bool], list[str]]:
    """Validate each SMILES in a list via RDKit parse.

    Returns (valid_flags, warnings).
    """
    from rdkit import Chem

    valid_flags: list[bool] = []
    warnings: list[str] = []

    for smi in smiles_list:
        if not smi or not smi.strip():
            valid_flags.append(False)
            warnings.append(f"Empty SMILES in list")
            continue

        lr_smi = smi.replace("[*]", "[Lr]")
        mol = Chem.MolFromSmiles(lr_smi, sanitize=False)
        if mol is None:
            valid_flags.append(False)
            warnings.append(f"RDKit parse failed: {smi}")
            continue

        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ),
            )
            valid_flags.append(True)
        except Exception:
            valid_flags.append(False)
            warnings.append(f"RDKit sanitize failed: {smi}")

    return valid_flags, warnings


# =====================================================================
# Issue log loader (P1 audit report)
# =====================================================================

def _load_issue_index(audit_path: Path) -> dict[str, list[dict]]:
    """Load P1 audit report and index issues by qmof_id for O(1) lookup.

    Returns {qmof_id: [issue_dict, ...]}. Each issue_dict has:
      field, severity, message, source (qmof_id stripped).
    """
    if not audit_path.exists():
        print(f"WARNING: Audit report not found at {audit_path}")
        return {}

    try:
        report = json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: Failed to load audit report: {e}")
        return {}

    index: dict[str, list[dict]] = {}
    for issue in report.get("issue_log", []):
        qid = issue.get("qmof_id", "")
        if not qid:
            continue
        if qid not in index:
            index[qid] = []
        index[qid].append(issue)

    return index


# =====================================================================
# SHA256 helper
# =====================================================================

def _file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================
# Main record builder
# =====================================================================

def build_record(
    row: pd.Series,
    row_idx: int,
    audit_issues: list[dict],
    csv_sha256: str,
) -> QMOFRecordV2:
    """Build a complete QMOFRecordV2 from a CSV row + source JSON.

    Steps:
      A. Parse CSV columns into layer1_facts
      B. Metal node enrichment (source JSON)
      C. SMARTS functional group detection (linker SMILES)
      D. Build Layer2Semantics
      E. Build Provenance
      F. Collect issue log entries
      G. Validate
      H. Assemble and return
    """
    all_issues: list[IssueLogEntry] = []

    # ── Step A: Parse CSV columns ──────────────────────────────────────
    qmof_id = str(row[COL["qmof_id"]])
    formula = _safe_str(row.get(COL["formula"], ""))

    smiles_nodes = _parse_smiles_list(row.get(COL["smiles_nodes"]))
    smiles_linkers = _parse_smiles_list(row.get(COL["smiles_linkers"]))

    # Validate SMILES
    nodes_valid, nodes_warnings = _validate_smiles_list(smiles_nodes)
    linkers_valid, linkers_warnings = _validate_smiles_list(smiles_linkers)
    smiles_warnings = nodes_warnings + linkers_warnings

    # ── Step B: Metal node enrichment ──────────────────────────────────
    metal_node, src_json_file, src_json_sha, metal_issues = enrich_metal_node(
        qmof_id, formula
    )
    all_issues.extend(metal_issues)

    # ── Step C: SMARTS functional group detection ──────────────────────
    per_linker, mof_fg, abstract_features, inferred_properties = enrich_mof_smarts(
        smiles_nodes, smiles_linkers
    )

    # Cross-check: update linkers_valid from per_linker results
    # (SMARTS engine may have stricter/different validation)
    if per_linker and len(per_linker) == len(smiles_linkers):
        linkers_valid = [le.smiles_valid for le in per_linker]

    # Merge open_metal_site from metal enricher into abstract_features
    if metal_node.has_open_metal_sites is not None:
        abstract_features.has_open_metal_site = metal_node.has_open_metal_sites

    # ── Step A (continued): Build Layer1Facts ──────────────────────────
    layer1 = Layer1Facts(
        formula=formula,
        natoms=_safe_int(row.get(COL["natoms"])),
        density=_safe_float(row.get(COL["density"])),
        volume=_safe_float(row.get(COL["volume"])),
        pld=_safe_float(row.get(COL["pld"])),
        lcd=_safe_float(row.get(COL["lcd"])),
        topology=_safe_str(row.get(COL["topology"], "")),
        spacegroup=_safe_str(row.get(COL["spacegroup"], "")),
        crystal_system=_safe_str(row.get(COL["crystal_system"], "")),
        mofid=_safe_str(row.get(COL["mofid"], "")),
        mofkey=_safe_str(row.get(COL["mofkey"], "")),
        synthesized=_safe_bool(row.get(COL["synthesized"])),
        doi=_safe_str(row.get(COL["doi"], "")),
        smiles_nodes=smiles_nodes,
        smiles_linkers=smiles_linkers,
        smiles_nodes_valid=nodes_valid,
        smiles_linkers_valid=linkers_valid,
        smiles_validation_warnings=smiles_warnings,
        bandgap_pbe=_safe_float(row.get(COL["bandgap_pbe"])),
        bandgap_hle17=_safe_float(row.get(COL["bandgap_hle17"])),
        bandgap_hse06_10hf=_safe_float(row.get(COL["bandgap_hse06_10hf"])),
        bandgap_hse06=_safe_float(row.get(COL["bandgap_hse06"])),
        metal_node=metal_node,
    )

    # ── Step D: Build Layer2Semantics ──────────────────────────────────
    layer2 = Layer2Semantics(
        source="smarts_rules",
        linker_enrichment=per_linker,
        functional_groups=mof_fg,
        abstract_features=abstract_features,
        coordinating_groups=metal_node.coordinating_groups,
        inferred_properties=inferred_properties,
        readable_name=None,     # filled by LLM enricher later
        design_hints=None,      # filled by LLM enricher later
    )

    # ── Step E: Build Provenance ───────────────────────────────────────
    provenance = make_provenance(
        source_csv_row=row_idx,
        source_json=src_json_file,
        source_json_sha256=src_json_sha,
        layer2_method="smarts_rules",
        smarts_pattern_count=SMARTS_PATTERN_COUNT,
    )
    # Store CSV SHA in field_sources for traceability
    provenance.field_sources["source_csv_sha256"] = csv_sha256

    # ── Step F: Collect issue log entries ──────────────────────────────
    # Add audit report issues for this record
    for ai in audit_issues:
        all_issues.append(IssueLogEntry(
            field=ai.get("field", ""),
            severity=ai.get("severity", "info"),
            message=ai.get("message", ""),
            source=ai.get("source"),
        ))

    # ── Step G+H: Assemble record, then validate ──────────────────────
    record = QMOFRecordV2(
        qmof_id=qmof_id,
        record_type="mof",
        layer1_facts=layer1,
        layer2_semantics=layer2,
        provenance=provenance,
        validation_report=ValidationReport(issue_log=all_issues),
    )

    # Validate (reads record, produces ValidationReport preserving issue_log)
    report = validate_record(record)
    record.validation_report = report

    return record


# =====================================================================
# Batch pipeline runner
# =====================================================================

def run_pipeline(
    csv_path: Path = SOURCE_CSV,
    output_dir: Path = OUTPUT_DIR,
    limit: Optional[int] = None,
    skip_existing: bool = False,
    progress_interval: int = 500,
) -> dict:
    """Run the full deterministic enrichment pipeline.

    Args:
        csv_path: Path to qmof.csv source file.
        output_dir: Directory for output JSONs (one per MOF).
        limit: Process only first N records (None = all).
        skip_existing: Skip records whose output JSON already exists.
        progress_interval: Print progress every N records.

    Returns:
        Summary dict with counts and timing.
    """
    t0 = time.time()

    # ── Setup ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline_v2] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    total_csv = len(df)
    print(f"[pipeline_v2] CSV loaded: {total_csv} records, {len(df.columns)} columns")

    if limit is not None:
        df = df.head(limit)
        print(f"[pipeline_v2] Limited to first {limit} records")

    # Load P1 audit report issues indexed by qmof_id
    print(f"[pipeline_v2] Loading audit report: {AUDIT_REPORT}")
    issue_index = _load_issue_index(AUDIT_REPORT)
    print(f"[pipeline_v2] Audit issues indexed: {len(issue_index)} records with issues")

    # Compute CSV SHA256 once
    csv_sha256 = _file_sha256(csv_path)
    print(f"[pipeline_v2] CSV SHA256: {csv_sha256[:16]}...")

    # ── Process records ────────────────────────────────────────────────
    success_count = 0
    error_count = 0
    skipped_count = 0
    error_ids: list[str] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        qmof_id = str(row["qmof_id"])
        out_path = output_dir / f"{qmof_id}.json"

        # Skip existing if requested
        if skip_existing and out_path.exists():
            skipped_count += 1
            continue

        try:
            # Get audit issues for this record
            audit_issues = issue_index.get(qmof_id, [])

            # Build record
            record = build_record(row, idx, audit_issues, csv_sha256)

            # Serialize and write
            json_str = json.dumps(
                record.to_dict(), indent=2, ensure_ascii=False
            )
            out_path.write_text(json_str, encoding="utf-8")

            success_count += 1

        except Exception as e:
            error_count += 1
            error_ids.append(qmof_id)
            print(f"  ERROR [{qmof_id}]: {type(e).__name__}: {e}")

        # Progress
        processed = idx + 1
        if processed % progress_interval == 0 or processed == len(df):
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            print(
                f"  [{processed}/{len(df)}] "
                f"ok={success_count} err={error_count} skip={skipped_count} "
                f"({rate:.1f} rec/s, {elapsed:.1f}s)"
            )

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = {
        "total_csv": total_csv,
        "processed": len(df),
        "success": success_count,
        "errors": error_count,
        "skipped": skipped_count,
        "error_ids": error_ids[:20],  # cap at 20 for readability
        "elapsed_seconds": round(elapsed, 2),
        "records_per_second": round(success_count / elapsed, 2) if elapsed > 0 else 0,
        "output_dir": str(output_dir),
    }

    print(f"\n{'='*60}")
    print(f"[pipeline_v2] DONE")
    print(f"  Total CSV rows:  {total_csv}")
    print(f"  Processed:       {len(df)}")
    print(f"  Success:         {success_count}")
    print(f"  Errors:          {error_count}")
    print(f"  Skipped:         {skipped_count}")
    print(f"  Elapsed:         {elapsed:.1f}s")
    print(f"  Rate:            {summary['records_per_second']} rec/s")
    print(f"  Output:          {output_dir}")
    if error_ids:
        print(f"  First errors:    {error_ids[:5]}")
    print(f"{'='*60}")

    return summary


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="QMOF v2 Enrichment Pipeline (deterministic pass)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N records"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip already-generated JSONs"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory"
    )
    parser.add_argument(
        "--progress", type=int, default=500,
        help="Progress print interval (default: 500)"
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    summary = run_pipeline(
        output_dir=out,
        limit=args.limit,
        skip_existing=args.skip_existing,
        progress_interval=args.progress,
    )
