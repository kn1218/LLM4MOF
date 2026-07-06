#!/usr/bin/env python
"""hMOF Enriched v1 → v2 Pipeline.

Converts 51,164 hMOF enriched v1 JSONs into v2 format matching QMOF v2
two-layer architecture:
  layer1_facts      — Deterministic data (physical properties, identifiers)
  layer2_semantics  — SMARTS-based functional group detection
  provenance        — Reproducibility metadata
  validation_report — 12 automated checks

Usage:
    python hMOF/hmof_pipeline_v2.py --limit 10
    python hMOF/hmof_pipeline_v2.py --skip-existing
    python hMOF/hmof_pipeline_v2.py
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict

# ── Path setup ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # database_construction
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "PORMAKE"))

from qmof.qmof_pipeline.smarts_engine import enrich_mof_smarts  # noqa: E402
from bb_pipeline.smarts_library import ALL_PATTERNS              # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────
PIPELINE_VERSION = "2.0.0"
SCHEMA_VERSION = "2.0.0"
SMARTS_PATTERN_COUNT = len(ALL_PATTERNS)

GAS_FIELDS = [
    "h2_uptake_2bar_77K",
    "h2_uptake_100bar_77K",
    "ch4_uptake_35bar_298K",
    "co2_uptake_2_5bar_298K",
    "xe_loading_1bar_273K",
    "kr_loading_1bar_273K",
    "xekr_selectivity_1bar",
]

# Tool versions (computed once at module load)
_TOOL_VERSIONS: dict[str, str] = {"python": sys.version.split()[0]}
try:
    import rdkit
    _TOOL_VERSIONS["rdkit"] = rdkit.__version__
except (ImportError, AttributeError):
    pass


# =====================================================================
# Helpers
# =====================================================================

def _extract_surface_area_m2g(sa_raw) -> float | None:
    """Extract surface area in m2/g from v1 format.

    v1 surface_area can be a plain float or a dict with 'm2g'/'m2cm3' keys.
    """
    if sa_raw is None:
        return None
    if isinstance(sa_raw, dict):
        return sa_raw.get("m2g") or sa_raw.get("m2/g")
    return float(sa_raw)


def _safe_get(d: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


# =====================================================================
# V1 → V2 Field Mapping
# =====================================================================

def map_v1_to_v2(v1: dict) -> dict:
    """Convert a single v1 JSON record into v2 structure.

    Performs:
      1. Field restructuring (v1 flat → v2 two-layer)
      2. SMARTS functional group detection via shared engine
      3. Validation checks
    """
    hmof_id = v1.get("hmof_id", "")
    source = v1.get("source", {})
    enrichment = v1.get("enrichment", {})
    phys = v1.get("physical_properties", {})
    gas_raw = v1.get("gas_adsorption", {})

    smiles_nodes = source.get("smiles_nodes", [])
    smiles_linkers = source.get("smiles_linkers", [])

    # -- SMARTS enrichment (the expensive step) --
    try:
        per_linker, mof_fg, abstract, properties = enrich_mof_smarts(
            smiles_nodes, smiles_linkers
        )
        # Convert dataclass results to plain dicts
        per_linker_dicts = [asdict(le) for le in per_linker]
        mof_fg_dict = asdict(mof_fg)
        abstract_dict = asdict(abstract)
        smarts_ok = True
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
        smarts_ok = False

    # -- Per-SMILES validation arrays --
    v1_nodes_valid = _safe_get(enrichment, "smiles_validation", "smiles_nodes_valid", default=False)
    v1_linkers_valid = _safe_get(enrichment, "smiles_validation", "smiles_linkers_valid", default=False)

    # For linkers: use per-linker validation from SMARTS engine (more accurate)
    if per_linker_dicts:
        smiles_linkers_valid = [le.get("smiles_valid", False) for le in per_linker_dicts]
    else:
        smiles_linkers_valid = [v1_linkers_valid] * len(smiles_linkers)

    # For nodes: broadcast v1 overall flag
    smiles_nodes_valid = [v1_nodes_valid] * len(smiles_nodes)

    # -- Gas adsorption (preserve all 7 fields, null if missing) --
    gas_adsorption = {}
    for field in GAS_FIELDS:
        gas_adsorption[field] = gas_raw.get(field)

    # -- Build v2 record --
    v2 = {
        "hmof_id": hmof_id,
        "record_type": "mof",
        "database": "hMOF",

        "layer1_facts": {
            "name": source.get("name", hmof_id),
            "formula": None,
            "topology": source.get("topology_from_mofid"),
            "density": phys.get("density"),
            "pld": phys.get("pore_limiting_diameter"),
            "lcd": phys.get("largest_cavity_diameter"),
            "surface_area_m2g": _extract_surface_area_m2g(phys.get("surface_area")),
            "void_fraction": phys.get("void_fraction"),
            "synthesized": False,
            "mofid": source.get("mofid"),
            "mofkey": source.get("mofkey"),
            "smiles_nodes": smiles_nodes,
            "smiles_linkers": smiles_linkers,
            "smiles_nodes_valid": smiles_nodes_valid,
            "smiles_linkers_valid": smiles_linkers_valid,
            "metal_node": {
                "metals": source.get("metals", []),
                "nuclearity": None,
                "connectivity": None,
                "geometry": None,
                "sbu_type": None,
                "oxidation_states": None,
                "has_open_metal_sites": None,
                "coordinating_groups": [],
                "ligand_chemistry": [],
            },
            "gas_adsorption": gas_adsorption,
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
            "source_database": "hMOF",
            "source_file": f"hmof_enriched_v1/{hmof_id}_enriched.json",
            "layer1_method": "v1_migration",
            "layer2_method": "smarts_rules",
            "smarts_pattern_count": SMARTS_PATTERN_COUNT,
            "tool_versions": _TOOL_VERSIONS.copy(),
        },
    }

    # -- Validation --
    v2["validation_report"] = validate_record(v2, smarts_ok)

    return v2


# =====================================================================
# Validation (12 checks)
# =====================================================================

def validate_record(v2: dict, smarts_ok: bool) -> dict:
    """Run 12 validation checks on a v2 record.

    Returns a validation_report dict with status, checks, warnings, errors.
    """
    l1 = v2.get("layer1_facts", {})
    l2 = v2.get("layer2_semantics", {})
    fg = l2.get("functional_groups", {})

    checks = {}
    warnings = []
    errors = []
    issue_log = []

    # 1. has_id
    has_id = bool(v2.get("hmof_id", "").strip())
    checks["has_id"] = has_id
    if not has_id:
        errors.append("missing hmof_id")

    # 2. has_smiles_nodes
    nodes = l1.get("smiles_nodes", [])
    checks["has_smiles_nodes"] = bool(nodes)
    if not nodes:
        warnings.append("no SMILES nodes")

    # 3. has_smiles_linkers
    linkers = l1.get("smiles_linkers", [])
    checks["has_smiles_linkers"] = bool(linkers)
    if not linkers:
        warnings.append("no SMILES linkers")

    # 4. smiles_nodes_valid
    nodes_valid = l1.get("smiles_nodes_valid", [])
    checks["smiles_nodes_valid"] = any(nodes_valid) if nodes_valid else False
    if nodes_valid and not any(nodes_valid):
        warnings.append("no valid node SMILES")

    # 5. smiles_linkers_valid
    linkers_valid = l1.get("smiles_linkers_valid", [])
    checks["smiles_linkers_valid"] = any(linkers_valid) if linkers_valid else False
    if linkers_valid and not any(linkers_valid):
        warnings.append("no valid linker SMILES")

    # 6. has_metals
    metals = l1.get("metal_node", {}).get("metals", [])
    checks["has_metals"] = bool(metals)
    if not metals:
        warnings.append("no metals listed")

    # 7. has_topology
    topo = l1.get("topology")
    checks["has_topology"] = bool(topo and topo != "UNKNOWN")
    if not topo or topo == "UNKNOWN":
        warnings.append("no topology information")

    # 8. density_positive
    density = l1.get("density")
    if density is not None:
        checks["density_positive"] = density > 0
        if density <= 0:
            errors.append(f"non-positive density: {density}")
    else:
        checks["density_positive"] = True  # no data = pass

    # 9. pld_positive
    pld = l1.get("pld")
    if pld is not None:
        checks["pld_positive"] = pld > 0
        if pld <= 0:
            warnings.append(f"non-positive PLD: {pld}")
    else:
        checks["pld_positive"] = True

    # 10. lcd_positive
    lcd = l1.get("lcd")
    if lcd is not None:
        checks["lcd_positive"] = lcd > 0
        if lcd <= 0:
            warnings.append(f"non-positive LCD: {lcd}")
    else:
        checks["lcd_positive"] = True

    # 11. fg_detection_ran
    rule_based = fg.get("rule_based", [])
    checks["fg_detection_ran"] = smarts_ok and bool(rule_based)
    if not smarts_ok:
        warnings.append("SMARTS detection failed")

    # 12. has_gas_data
    gas = l1.get("gas_adsorption", {})
    has_gas = any(gas.get(f) is not None for f in GAS_FIELDS)
    checks["has_gas_data"] = has_gas
    if not has_gas:
        warnings.append("no gas adsorption data")

    # Build issue_log from warnings and errors
    for w in warnings:
        issue_log.append({"field": "", "severity": "warning", "message": w, "source": "validation"})
    for e in errors:
        issue_log.append({"field": "", "severity": "error", "message": e, "source": "validation"})

    # Compute status
    if errors:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "checks": checks,
        "warnings": len(warnings),
        "errors": len(errors),
        "issue_log": issue_log,
    }


# =====================================================================
# Pipeline Runner
# =====================================================================

def run_pipeline(
    v1_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
    progress_interval: int = 2000,
):
    """Convert all hMOF v1 JSONs to v2 format.

    Parameters
    ----------
    v1_dir : Path
        Input directory with v1 JSONs. Default: hMOF/hmof_enriched_v1/
    output_dir : Path
        Output directory for v2 JSONs. Default: hMOF/hmof_enriched_v2/
    limit : int or None
        Max records to process (None = all).
    skip_existing : bool
        If True, skip records that already have a v2 JSON.
    progress_interval : int
        Print progress every N records.
    """
    if v1_dir is None:
        v1_dir = BASE_DIR / "hMOF" / "hmof_enriched_v1"
    else:
        v1_dir = Path(v1_dir)

    if output_dir is None:
        output_dir = BASE_DIR / "hMOF" / "hmof_enriched_v2"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover v1 files (skip files starting with _)
    v1_files = sorted([
        f for f in v1_dir.glob("*.json")
        if not f.name.startswith("_")
    ])

    total_available = len(v1_files)
    if limit is not None:
        v1_files = v1_files[:limit]

    total = len(v1_files)
    print(f"hMOF v1→v2 Pipeline")
    print(f"  Input:  {v1_dir} ({total_available} files)")
    print(f"  Output: {output_dir}")
    print(f"  Processing: {total} records" + (f" (limit={limit})" if limit else ""))
    print(f"  SMARTS patterns: {SMARTS_PATTERN_COUNT}")
    print(f"  Skip existing: {skip_existing}")
    print()

    t0 = time.time()
    processed = 0
    skipped = 0
    errors = 0
    status_counts = {"pass": 0, "warning": 0, "error": 0}

    for i, v1_path in enumerate(v1_files):
        # Derive output path: {hmof_id}.json
        stem = v1_path.stem.replace("_enriched", "")
        out_path = output_dir / f"{stem}.json"

        if skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            with open(v1_path, "r", encoding="utf-8") as f:
                v1_data = json.load(f)

            v2_data = map_v1_to_v2(v1_data)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(v2_data, f, indent=2, ensure_ascii=False)

            status = v2_data.get("validation_report", {}).get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            processed += 1

        except Exception as e:
            errors += 1
            print(f"  ERROR [{stem}]: {e}")

        # Progress reporting
        done = i + 1
        if done % progress_interval == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  [{done:>6}/{total}] "
                f"{elapsed:6.1f}s elapsed, "
                f"{rate:5.1f} rec/s, "
                f"ETA {eta:5.0f}s | "
                f"pass={status_counts['pass']} "
                f"warn={status_counts['warning']} "
                f"err={status_counts['error']}"
            )

    elapsed = time.time() - t0
    print()
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Status: pass={status_counts['pass']} "
          f"warning={status_counts['warning']} "
          f"error={status_counts['error']}")
    print(f"  Rate: {processed / elapsed:.1f} rec/s" if elapsed > 0 else "")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="hMOF v1→v2 enrichment pipeline"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max records to process (default: all)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip records that already have v2 output"
    )
    parser.add_argument(
        "--progress", type=int, default=2000,
        help="Progress report interval (default: 2000)"
    )
    args = parser.parse_args()

    run_pipeline(
        limit=args.limit,
        skip_existing=args.skip_existing,
        progress_interval=args.progress,
    )
