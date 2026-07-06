"""
source_investigator.py — Comprehensive audit of ALL QMOF source data.

Reads the full qmof.csv (20,373 records), validates every SMILES with RDKit,
cross-checks against source JSON files, and produces a structured issue report.

Usage:
    python -m qmof.qmof_pipeline.source_investigator
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # database_construction
SOURCE_DATA = BASE_DIR / "qmof" / "_source_data"
CSV_PATH = SOURCE_DATA / "qmof.csv"
JSON_DIR = SOURCE_DATA / "qmof_global_jsons_v2"
AUDIT_DIR = BASE_DIR / "qmof" / "_audit"
REPORT_PATH = AUDIT_DIR / "source_investigation_report.json"

PROGRESS_INTERVAL = 2000

# Canonical list of transition/lanthanide/actinide metals (periodic table)
METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
}

# Regex for element symbols in a formula string (e.g. "Cu3C9H14I4NS")
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")


# ===================================================================
# 1. CSV Profiling
# ===================================================================
def profile_csv(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """For each column: dtype, null_count, null_pct, unique_count, sample_values."""
    print("[1/5] Profiling CSV columns ...")
    profile: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        total = len(df)
        sample_vals = df[col].dropna().head(3).tolist()
        # Make sure sample values are JSON-serialisable
        sample_vals = [str(v) for v in sample_vals]
        profile[col] = {
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_count / total * 100, 2) if total else 0.0,
            "unique_count": int(df[col].nunique()),
            "sample_values": sample_vals,
        }
    print(f"       → profiled {len(profile)} columns")
    return profile


# ===================================================================
# 2. SMILES Validation (RDKit)
# ===================================================================
def _parse_smiles_list(raw: Any) -> list[str] | None:
    """Parse a Python list-string from CSV into a real list of SMILES.

    Returns None if the cell is NaN/empty, or an empty list if parsing fails.
    """
    if pd.isna(raw) or str(raw).strip() in ("", "nan"):
        return None
    try:
        parsed = ast.literal_eval(str(raw))
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
    except (ValueError, SyntaxError):
        pass
    return None


def _validate_one_smiles(smi: str) -> tuple[bool, str]:
    """Validate a single SMILES using RDKit.

    Returns (valid: bool, warning: str). Empty warning means valid.
    Uses Chem.MolFromSmiles with sanitize=False, then manual SanitizeMol
    with SANITIZE_ALL ^ SANITIZE_SETAROMATICITY to avoid kekulisation
    failures on metal-containing SMILES.
    """
    from rdkit import Chem
    from rdkit.Chem import SanitizeFlags

    smi = smi.strip()
    if not smi:
        return False, "empty_smiles_string"
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return False, "rdkit_parse_failed"
        # Sanitise everything except kekulise (which fails on metal atoms)
        san_flags = SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE
        Chem.SanitizeMol(mol, san_flags)
        return True, ""
    except Exception as exc:
        return False, str(exc)[:120]


def validate_smiles(
    df: pd.DataFrame,
    issue_log: list[dict],
) -> dict[str, Any]:
    """Validate all SMILES in nodes + linkers columns. Returns summary dict."""
    print("[2/5] Validating SMILES (nodes + linkers) ...")
    stats: dict[str, Counter] = {
        "nodes": Counter(),
        "linkers": Counter(),
    }
    warning_counter: Counter = Counter()
    total = len(df)

    for idx, row in df.iterrows():
        if (idx + 1) % PROGRESS_INTERVAL == 0:
            print(f"       SMILES progress: {idx + 1}/{total}")

        qmof_id = row["qmof_id"]

        for field_key, col_name in [
            ("nodes", "info.mofid.smiles_nodes"),
            ("linkers", "info.mofid.smiles_linkers"),
        ]:
            raw_val = row.get(col_name)
            smi_list = _parse_smiles_list(raw_val)

            if smi_list is None:
                stats[field_key]["total_empty"] += 1
                issue_log.append({
                    "qmof_id": qmof_id,
                    "field": f"smiles_{field_key}",
                    "severity": "warning",
                    "message": "Empty / NaN SMILES list",
                    "source": "csv",
                })
                continue

            if len(smi_list) == 0:
                stats[field_key]["total_empty"] += 1
                issue_log.append({
                    "qmof_id": qmof_id,
                    "field": f"smiles_{field_key}",
                    "severity": "warning",
                    "message": "Parsed to empty list",
                    "source": "csv",
                })
                continue

            for si, smi in enumerate(smi_list):
                valid, warning = _validate_one_smiles(smi)
                if valid:
                    stats[field_key]["total_valid"] += 1
                else:
                    stats[field_key]["total_invalid"] += 1
                    warning_counter[warning] += 1
                    issue_log.append({
                        "qmof_id": qmof_id,
                        "field": f"smiles_{field_key}",
                        "severity": "error",
                        "message": f"Invalid SMILES idx={si}: {warning}",
                        "source": "csv",
                    })

    # Build top warnings
    top_warnings = [
        {"warning": w, "count": c}
        for w, c in warning_counter.most_common(10)
    ]
    summary = {
        "nodes": {
            "total_valid": stats["nodes"]["total_valid"],
            "total_invalid": stats["nodes"]["total_invalid"],
            "total_empty": stats["nodes"]["total_empty"],
        },
        "linkers": {
            "total_valid": stats["linkers"]["total_valid"],
            "total_invalid": stats["linkers"]["total_invalid"],
            "total_empty": stats["linkers"]["total_empty"],
        },
        "top_warnings": top_warnings,
    }
    print(f"       → nodes valid={summary['nodes']['total_valid']} "
          f"invalid={summary['nodes']['total_invalid']} "
          f"empty={summary['nodes']['total_empty']}")
    print(f"       → linkers valid={summary['linkers']['total_valid']} "
          f"invalid={summary['linkers']['total_invalid']} "
          f"empty={summary['linkers']['total_empty']}")
    return summary


# ===================================================================
# 3. Source JSON Matching
# ===================================================================
def match_source_jsons(
    df: pd.DataFrame,
    issue_log: list[dict],
) -> dict[str, Any]:
    """Check every CSV qmof_id against its source JSON file."""
    print("[3/5] Cross-checking CSV IDs vs source JSONs ...")
    total_matched = 0
    total_unmatched = 0
    total_with_metal_data = 0
    unmatched_ids: list[str] = []

    # Cache: read JSON only if it exists
    json_data_cache: dict[str, dict] = {}
    total = len(df)

    for idx, row in df.iterrows():
        if (idx + 1) % PROGRESS_INTERVAL == 0:
            print(f"       JSON matching progress: {idx + 1}/{total}")

        qmof_id: str = row["qmof_id"]
        json_path = JSON_DIR / f"{qmof_id}_analysis.json"

        if json_path.exists():
            total_matched += 1
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    jdata = json.load(fh)
                json_data_cache[qmof_id] = jdata
                mn = jdata.get("metal_node")
                if mn and isinstance(mn, dict):
                    comp = mn.get("composition", {})
                    metals = comp.get("metals", [])
                    if metals:
                        total_with_metal_data += 1
                    else:
                        issue_log.append({
                            "qmof_id": qmof_id,
                            "field": "metal_node.composition.metals",
                            "severity": "warning",
                            "message": "JSON exists but metals list is empty",
                            "source": "source_json",
                        })
                else:
                    issue_log.append({
                        "qmof_id": qmof_id,
                        "field": "metal_node",
                        "severity": "warning",
                        "message": "JSON exists but metal_node key missing or invalid",
                        "source": "source_json",
                    })
            except (json.JSONDecodeError, OSError) as exc:
                issue_log.append({
                    "qmof_id": qmof_id,
                    "field": "source_json",
                    "severity": "error",
                    "message": f"Failed to read JSON: {exc}",
                    "source": "source_json",
                })
        else:
            total_unmatched += 1
            unmatched_ids.append(qmof_id)
            issue_log.append({
                "qmof_id": qmof_id,
                "field": "source_json",
                "severity": "info",
                "message": "No matching source JSON file",
                "source": "source_json",
            })

    summary = {
        "total_matched": total_matched,
        "total_unmatched": total_unmatched,
        "total_with_metal_data": total_with_metal_data,
        "unmatched_ids": unmatched_ids,
    }
    print(f"       → matched={total_matched}  unmatched={total_unmatched}  "
          f"with_metal_data={total_with_metal_data}")
    return summary, json_data_cache


# ===================================================================
# 4. Cross-Validation (metals in CSV formula vs JSON)
# ===================================================================
def _extract_metals_from_formula(formula: Any) -> set[str]:
    """Extract metal element symbols from a chemical formula string."""
    if pd.isna(formula) or not str(formula).strip():
        return set()
    elements = _ELEMENT_RE.findall(str(formula))
    return {e for e in elements if e in METALS}


def cross_validate_metals(
    df: pd.DataFrame,
    json_data_cache: dict[str, dict],
    issue_log: list[dict],
) -> dict[str, Any]:
    """Compare metals in CSV formula vs source JSON metal_node."""
    print("[4/5] Cross-validating metals (CSV formula vs JSON) ...")
    discrepancies: list[dict] = []
    total_checked = 0
    total_consistent = 0
    total_inconsistent = 0

    for idx, row in df.iterrows():
        qmof_id: str = row["qmof_id"]
        if qmof_id not in json_data_cache:
            continue

        jdata = json_data_cache[qmof_id]
        mn = jdata.get("metal_node")
        if not mn or not isinstance(mn, dict):
            continue

        json_metals_raw = mn.get("composition", {}).get("metals", [])
        if not json_metals_raw:
            continue

        total_checked += 1
        csv_metals = _extract_metals_from_formula(row.get("info.formula", ""))
        json_metals = set(json_metals_raw)

        if csv_metals == json_metals:
            total_consistent += 1
        else:
            total_inconsistent += 1
            disc = {
                "qmof_id": qmof_id,
                "csv_metals": sorted(csv_metals),
                "json_metals": sorted(json_metals),
                "csv_only": sorted(csv_metals - json_metals),
                "json_only": sorted(json_metals - csv_metals),
            }
            discrepancies.append(disc)
            issue_log.append({
                "qmof_id": qmof_id,
                "field": "metals",
                "severity": "warning",
                "message": (
                    f"Metal mismatch: CSV={sorted(csv_metals)} "
                    f"JSON={sorted(json_metals)}"
                ),
                "source": "cross_validation",
            })

    summary = {
        "metal_discrepancies": discrepancies,
        "total_checked": total_checked,
        "total_consistent": total_consistent,
        "total_inconsistent": total_inconsistent,
    }
    print(f"       → checked={total_checked}  consistent={total_consistent}  "
          f"inconsistent={total_inconsistent}")
    return summary


# ===================================================================
# 5. Issue Summary
# ===================================================================
def summarise_issues(issue_log: list[dict]) -> dict[str, Any]:
    """Build aggregate issue summary."""
    by_severity: Counter = Counter()
    by_field: Counter = Counter()
    for issue in issue_log:
        by_severity[issue["severity"]] += 1
        by_field[issue["field"]] += 1
    return {
        "total_issues": len(issue_log),
        "by_severity": dict(by_severity),
        "by_field": dict(by_field),
    }


# ===================================================================
# Main pipeline
# ===================================================================
def run_investigation() -> dict[str, Any]:
    """Execute the full source investigation pipeline."""
    t0 = time.time()
    print("=" * 60)
    print("QMOF Source Investigation -- Full Audit")
    print("=" * 60)

    # --- Load CSV ---
    print(f"\nLoading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"Loaded {len(df)} records x {len(df.columns)} columns\n")

    issue_log: list[dict] = []

    # --- 1. CSV Profile ---
    csv_profile = profile_csv(df)

    # --- 2. SMILES Validation ---
    smiles_summary = validate_smiles(df, issue_log)

    # --- 3. Source JSON Matching ---
    json_summary, json_cache = match_source_jsons(df, issue_log)

    # --- 4. Cross-Validation ---
    cross_summary = cross_validate_metals(df, json_cache, issue_log)

    # --- 5. Issue Summary ---
    issue_summary = summarise_issues(issue_log)

    elapsed = round(time.time() - t0, 1)
    print(f"\n[5/5] Building report ... ({elapsed}s elapsed)")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(df),
        "csv_profile": csv_profile,
        "smiles_validation": smiles_summary,
        "source_json_matching": json_summary,
        "cross_validation": cross_summary,
        "issue_log": issue_log,
        "issue_summary": issue_summary,
        "elapsed_seconds": elapsed,
    }

    # --- Write report ---
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] Report written to {REPORT_PATH}")
    print(f"  File size: {REPORT_PATH.stat().st_size / 1024:.1f} KB")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total records audited : {report['total_records']}")
    print(f"  Total issues found    : {issue_summary['total_issues']}")
    print(f"    errors              : {issue_summary['by_severity'].get('error', 0)}")
    print(f"    warnings            : {issue_summary['by_severity'].get('warning', 0)}")
    print(f"    info                : {issue_summary['by_severity'].get('info', 0)}")
    print(f"  SMILES nodes   valid={smiles_summary['nodes']['total_valid']}  "
          f"invalid={smiles_summary['nodes']['total_invalid']}  "
          f"empty={smiles_summary['nodes']['total_empty']}")
    print(f"  SMILES linkers valid={smiles_summary['linkers']['total_valid']}  "
          f"invalid={smiles_summary['linkers']['total_invalid']}  "
          f"empty={smiles_summary['linkers']['total_empty']}")
    print(f"  Source JSONs matched  : {json_summary['total_matched']}")
    print(f"  Source JSONs missing  : {json_summary['total_unmatched']}")
    print(f"  Metal cross-checks    : {cross_summary['total_checked']}  "
          f"(consistent={cross_summary['total_consistent']}  "
          f"inconsistent={cross_summary['total_inconsistent']})")
    print(f"  Elapsed time          : {elapsed}s")
    print("=" * 60)

    return report


# ===================================================================
# Entry-point
# ===================================================================
if __name__ == "__main__":
    run_investigation()
