"""CoRE-MOF v1 → v2 enrichment pipeline.

Converts CoRE-MOF enriched v1 JSONs into v2 format matching QMOF v2 structure.
Performs SMARTS-based functional group detection via shared engine, validation,
and outputs one v2 JSON per record.

Usage:
    python CoRE-MOF/coremof_pipeline_v2.py --limit 5
    python CoRE-MOF/coremof_pipeline_v2.py --skip-existing --progress 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Path setup ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # database_construction
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "PORMAKE"))

from qmof.qmof_pipeline.smarts_engine import enrich_mof_smarts  # noqa: E402
from bb_pipeline.smarts_library import ALL_PATTERNS  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────
PIPELINE_VERSION = "2.0.0"
SCHEMA_VERSION = "2.0.0"

V1_DIR = BASE_DIR / "CoRE-MOF" / "coremof_enriched_v1"
V2_DIR = BASE_DIR / "CoRE-MOF" / "coremof_enriched_v2"
ASR_CSV = BASE_DIR / "CoRE-MOF" / "_source_data" / "ASR_data_SI_20250204.csv"


# =====================================================================
# Tool version detection
# =====================================================================

def _get_tool_versions() -> dict[str, str]:
    """Collect runtime tool versions."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    try:
        import rdkit
        versions["rdkit"] = rdkit.__version__
    except (ImportError, AttributeError):
        pass
    try:
        import pandas
        versions["pandas"] = pandas.__version__
    except (ImportError, AttributeError):
        pass
    return versions


TOOL_VERSIONS = _get_tool_versions()


# =====================================================================
# ASR CSV loader (optional supplementary data)
# =====================================================================

def load_asr_csv(csv_path: Path) -> dict[str, dict]:
    """Load ASR CSV into {coreid: row_dict} lookup.

    Returns empty dict if file is missing or unreadable.
    """
    if not csv_path.exists():
        print(f"[WARN] ASR CSV not found: {csv_path}")
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        lookup: dict[str, dict] = {}
        for _, row in df.iterrows():
            cid = str(row.get("coreid", "")).strip()
            if cid:
                lookup[cid] = row.to_dict()
        print(f"[INFO] Loaded ASR CSV: {len(lookup)} records")
        return lookup
    except Exception as e:
        print(f"[WARN] Failed to load ASR CSV: {e}")
        return {}


# =====================================================================
# Safe getters
# =====================================================================

def _safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Nested dict lookup with fallback."""
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
        else:
            return default
    return current


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float or None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Convert to int or None."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# =====================================================================
# Topology resolver
# =====================================================================

def _resolve_topology(v1: dict) -> Optional[str]:
    """Pick best topology: mofid > csv, skip 'nan'/'unknown'/'error'/empty."""
    invalid = {"", "nan", "unknown", "unnamed", "error"}
    for key in ("topology_from_mofid", "topology_from_csv"):
        val = _safe_get(v1, "source", key)
        if val and str(val).strip().lower() not in invalid:
            return str(val).strip()
    return None


# =====================================================================
# V1 → V2 field mapping
# =====================================================================

def map_v1_to_v2(v1: dict, asr_row: Optional[dict] = None) -> dict:
    """Convert a single v1 record to v2 format (plain dict).

    Parameters
    ----------
    v1 : dict
        Parsed v1 enriched JSON.
    asr_row : dict, optional
        Matching row from ASR CSV for supplementary data.

    Returns
    -------
    dict
        V2-format record ready for JSON serialization.
    """
    coremof_id = v1.get("coremof_id", "")
    source = v1.get("source", {})
    enrichment = v1.get("enrichment", {})
    phys = v1.get("physical_properties", {})
    stab = v1.get("stability", {})

    # SMILES
    smiles_nodes = source.get("smiles_nodes", []) or []
    smiles_linkers = source.get("smiles_linkers", []) or []

    # SMILES validation from v1
    smiles_val = enrichment.get("smiles_validation", {})
    nodes_valid_flag = smiles_val.get("smiles_nodes_valid", False)
    linkers_valid_flag = smiles_val.get("smiles_linkers_valid", False)

    # Per-SMILES valid lists (v2 expects per-item booleans)
    smiles_nodes_valid = [nodes_valid_flag] * len(smiles_nodes) if smiles_nodes else []
    smiles_linkers_valid = [linkers_valid_flag] * len(smiles_linkers) if smiles_linkers else []

    # Topology
    topology = _resolve_topology(v1)

    # Metals
    metals = source.get("metals", []) or []

    # ── Run SMARTS enrichment ──────────────────────────────────────
    try:
        per_linker, mof_fg, abstract, properties = enrich_mof_smarts(
            smiles_nodes, smiles_linkers
        )
        per_linker_dicts = [asdict(le) for le in per_linker]
        mof_fg_dict = asdict(mof_fg)
        abstract_dict = asdict(abstract)
    except Exception as e:
        per_linker_dicts = []
        mof_fg_dict = {
            "backbone": [], "substituents": [], "rule_based": [],
            "rule_based_counts": {}, "llm_additions": [],
        }
        abstract_dict = {
            "is_fluorinated": None, "is_electron_deficient": None,
            "is_electron_rich": None, "is_symmetric": None,
            "is_conjugated": None, "is_metalated": None,
            "has_hydrogen_bond_donor": None, "has_hydrogen_bond_acceptor": None,
            "is_charged": None, "is_photoswitchable": None,
            "has_open_metal_site": None,
        }
        properties = []
        print(f"  [WARN] SMARTS enrichment failed for {coremof_id}: {e}")

    # Override abstract has_open_metal_site from stability data
    oms = stab.get("has_open_metal_sites")
    if oms is not None:
        abstract_dict["has_open_metal_site"] = bool(oms)

    # ── Build v2 record ────────────────────────────────────────────
    v2 = {
        "coremof_id": coremof_id,
        "record_type": "mof",
        "database": "CoRE-MOF",
        "layer1_facts": {
            "formula": source.get("formula"),
            "topology": topology,
            "density": _safe_float(phys.get("density")),
            "volume": None,
            "pld": _safe_float(phys.get("pore_limiting_diameter")),
            "lcd": _safe_float(phys.get("largest_cavity_diameter")),
            "surface_area_m2g": _safe_float(phys.get("surface_area")),
            "void_fraction": _safe_float(phys.get("void_fraction")),
            "natoms": _safe_int(phys.get("natoms")),
            "spacegroup_number": _safe_int(phys.get("spacegroup_number")),
            "crystal_system": phys.get("crystal_system"),
            "catenation": _safe_int(phys.get("catenation")),
            "synthesized": True,
            "doi": source.get("doi"),
            "mofid": source.get("mofid_v1"),
            "refcode": source.get("refcode"),
            "smiles_nodes": smiles_nodes,
            "smiles_linkers": smiles_linkers,
            "smiles_nodes_valid": smiles_nodes_valid,
            "smiles_linkers_valid": smiles_linkers_valid,
            "metal_node": {
                "metals": metals,
                "nuclearity": None,
                "connectivity": None,
                "geometry": None,
                "sbu_type": None,
                "oxidation_states": None,
                "has_open_metal_sites": stab.get("has_open_metal_sites"),
                "coordinating_groups": [],
                "ligand_chemistry": [],
            },
            "stability": {
                "thermal_stability_C": _safe_float(stab.get("thermal_stability_C")),
                "water_stability": _safe_float(stab.get("water_stability")),
                "solvent_stability": _safe_float(stab.get("solvent_stability")),
                "oms_types": stab.get("oms_types"),
            },
        },
        "layer2_semantics": {
            "source": "smarts_rules",
            "readable_name": None,
            "design_hints": None,
            "functional_groups": mof_fg_dict,
            "abstract_features": abstract_dict,
            "inferred_properties": properties,
            "linker_enrichment": per_linker_dicts,
        },
        "provenance": {
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_database": "CoRE-MOF",
            "source_file": f"coremof_enriched_v1/{coremof_id}_enriched.json",
            "layer1_method": "v1_migration+csv",
            "layer2_method": "smarts_rules",
            "smarts_pattern_count": len(ALL_PATTERNS),
            "tool_versions": TOOL_VERSIONS,
        },
        "validation_report": None,  # filled by validate()
    }

    return v2


# =====================================================================
# Validation
# =====================================================================

def validate_record(v2: dict) -> dict:
    """Run 12 validation checks and produce a validation_report dict.

    Each check produces {"name": ..., "status": ..., "detail": ...}.
    Overall status: "error" if any error, "warning" if any warning, else "pass".
    """
    l1 = v2.get("layer1_facts", {})
    l2 = v2.get("layer2_semantics", {})
    checks: list[dict] = []
    warnings = 0
    errors = 0

    def _add(name: str, ok: bool, severity: str, detail: str):
        nonlocal warnings, errors
        status = "pass" if ok else severity
        if not ok:
            if severity == "error":
                errors += 1
            else:
                warnings += 1
        checks.append({"name": name, "status": status, "detail": detail})

    # 1. has_id
    cid = v2.get("coremof_id", "")
    _add("has_id", bool(cid), "error", f"coremof_id={'present' if cid else 'MISSING'}")

    # 2. has_smiles_nodes
    sn = l1.get("smiles_nodes", [])
    _add("has_smiles_nodes", bool(sn), "warning", f"count={len(sn)}")

    # 3. has_smiles_linkers
    sl = l1.get("smiles_linkers", [])
    _add("has_smiles_linkers", bool(sl), "warning", f"count={len(sl)}")

    # 4. smiles_nodes_valid
    snv = l1.get("smiles_nodes_valid", [])
    ok = any(snv) if snv else False
    _add("smiles_nodes_valid", ok, "warning",
         f"valid={sum(snv) if snv else 0}/{len(snv)}")

    # 5. smiles_linkers_valid
    slv = l1.get("smiles_linkers_valid", [])
    ok = any(slv) if slv else False
    _add("smiles_linkers_valid", ok, "warning",
         f"valid={sum(slv) if slv else 0}/{len(slv)}")

    # 6. has_metals
    metals = l1.get("metal_node", {}).get("metals", [])
    _add("has_metals", bool(metals), "warning", f"metals={metals}")

    # 7. has_topology
    topo = l1.get("topology")
    ok = bool(topo) and str(topo).upper() not in ("UNKNOWN", "ERROR")
    _add("has_topology", ok, "warning", f"topology={topo}")

    # 8. density_positive
    density = l1.get("density")
    if density is not None:
        _add("density_positive", density > 0, "error", f"density={density}")
    else:
        _add("density_positive", False, "warning", "density=None")

    # 9. pld_positive
    pld = l1.get("pld")
    if pld is not None:
        _add("pld_positive", pld > 0, "warning", f"pld={pld}")
    else:
        _add("pld_positive", False, "warning", "pld=None")

    # 10. lcd_positive
    lcd = l1.get("lcd")
    if lcd is not None:
        _add("lcd_positive", lcd > 0, "warning", f"lcd={lcd}")
    else:
        _add("lcd_positive", False, "warning", "lcd=None")

    # 11. fg_detection_ran
    fg = l2.get("functional_groups", {})
    rb = fg.get("rule_based", [])
    _add("fg_detection_ran", bool(rb), "warning", f"rule_based_count={len(rb)}")

    # 12. has_doi
    doi = l1.get("doi")
    _add("has_doi", bool(doi), "warning", f"doi={'present' if doi else 'MISSING'}")

    # Issue log from warnings in v1 SMILES validation (preserve original issues)
    issue_log: list[dict] = []

    # Overall status
    if errors > 0:
        status = "error"
    elif warnings > 0:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "issue_log": issue_log,
    }


# =====================================================================
# Safe filename helper
# =====================================================================

def _safe_filename(coremof_id: str) -> str:
    """Convert coremof_id to a safe filename (keep as-is, v1 files use same scheme)."""
    return coremof_id


# =====================================================================
# Pipeline runner
# =====================================================================

def run_pipeline(
    v1_dir: Path = V1_DIR,
    output_dir: Path = V2_DIR,
    limit: Optional[int] = None,
    skip_existing: bool = False,
    progress_interval: int = 100,
) -> None:
    """Convert all v1 JSONs to v2 format.

    Parameters
    ----------
    v1_dir : Path
        Directory containing v1 enriched JSONs.
    output_dir : Path
        Output directory for v2 JSONs.
    limit : int, optional
        Process only first N records (for testing).
    skip_existing : bool
        If True, skip records whose output file already exists.
    progress_interval : int
        Print progress every N records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ASR CSV for supplementary data
    asr_lookup = load_asr_csv(ASR_CSV)

    # Collect v1 JSON files (skip report files)
    v1_files = sorted([
        f for f in v1_dir.glob("*.json")
        if not f.name.startswith("_")
    ])

    total = len(v1_files)
    if limit is not None:
        v1_files = v1_files[:limit]
    to_process = len(v1_files)

    print(f"\n{'='*60}")
    print(f"CoRE-MOF v1 → v2 Pipeline")
    print(f"{'='*60}")
    print(f"  Total v1 files found: {total}")
    print(f"  Processing:           {to_process}")
    print(f"  Output dir:           {output_dir}")
    print(f"  Skip existing:        {skip_existing}")
    print(f"  SMARTS patterns:      {len(ALL_PATTERNS)}")
    print(f"{'='*60}\n")

    # Counters
    processed = 0
    skipped = 0
    status_counts = {"pass": 0, "warning": 0, "error": 0}
    t0 = time.time()

    for i, v1_path in enumerate(v1_files):
        # Load v1 JSON
        try:
            with open(v1_path, "r", encoding="utf-8") as f:
                v1_data = json.load(f)
        except Exception as e:
            print(f"  [ERROR] Failed to read {v1_path.name}: {e}")
            continue

        coremof_id = v1_data.get("coremof_id", v1_path.stem.replace("_enriched", ""))

        # Output file
        out_filename = f"{_safe_filename(coremof_id)}.json"
        out_path = output_dir / out_filename

        # Skip existing
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        # Map v1 → v2
        asr_row = asr_lookup.get(coremof_id)
        v2_record = map_v1_to_v2(v1_data, asr_row)

        # Validate
        v2_record["validation_report"] = validate_record(v2_record)

        # Write output
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(v2_record, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [ERROR] Failed to write {out_path.name}: {e}")
            continue

        processed += 1
        status = v2_record["validation_report"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

        # Progress
        if processed % progress_interval == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  [{processed}/{to_process}] {rate:.1f} rec/s | "
                  f"pass={status_counts['pass']} warn={status_counts['warning']} "
                  f"err={status_counts['error']}")

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Processed:  {processed}")
    print(f"  Skipped:    {skipped}")
    print(f"  Status:     pass={status_counts['pass']}  "
          f"warning={status_counts['warning']}  error={status_counts['error']}")
    print(f"  Time:       {elapsed:.1f}s ({processed/elapsed:.1f} rec/s)"
          if elapsed > 0 else "  Time:       <1s")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CoRE-MOF v1 → v2 enrichment pipeline"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N records (for testing)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip records whose output file already exists"
    )
    parser.add_argument(
        "--progress", type=int, default=100,
        help="Print progress every N records (default: 100)"
    )
    args = parser.parse_args()

    run_pipeline(
        limit=args.limit,
        skip_existing=args.skip_existing,
        progress_interval=args.progress,
    )
