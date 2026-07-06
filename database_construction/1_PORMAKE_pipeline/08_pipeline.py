"""Phase 7: Pipeline assembler — runs the full metadata generation pipeline.

Usage:
    python -m bb_pipeline.pipeline          # Process all 867 BBs
    python -m bb_pipeline.pipeline E1 N100  # Process specific BBs
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import rdkit
import networkx as nx

from .config import (
    BBS_DIR,
    OUTPUT_DIR,
    SCHEMA_VERSION,
    PIPELINE_VERSION,
)
from .xyz_parser import parse_xyz, parse_all_xyz, XYZParseError
from .feature_extractor import extract_layer1
from .smarts_enricher import enrich_layer2
from .node_enricher import enrich_node_fields
from .schema import (
    BBRecord,
    BBType,
    Provenance,
    make_pending_layer2,
)
from .validator import validate_record


def _sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()


def _tool_versions() -> dict[str, str]:
    """Collect versions of key tools used in the pipeline."""
    versions = {
        "rdkit": rdkit.__version__,
        "networkx": nx.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import selfies as sf
        versions["selfies"] = sf.__version__
    except ImportError:
        versions["selfies"] = "not_installed"
    return versions


def process_single(filepath: Path) -> BBRecord:
    """Process a single XYZ file into a complete BBRecord.

    Args:
        filepath: Path to a .xyz file (e.g., E1.xyz or N100.xyz).

    Returns:
        BBRecord with Layer 1 facts, pending Layer 2, provenance, validation.
    """
    # Parse XYZ
    parsed = parse_xyz(filepath)

    # Extract Layer 1 facts
    layer1 = extract_layer1(parsed)

    # Layer 2: pending (to be filled by SMARTS rules or LLM later)
    layer2 = make_pending_layer2()

    # Provenance
    provenance = Provenance(
        pipeline_version=PIPELINE_VERSION,
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_file=filepath.name,
        source_file_sha256=_sha256(filepath),
        tool_versions=_tool_versions(),
        dummy_atom_placeholder="Lr",
        layer1_method="deterministic",
        layer2_method="pending",
    )

    # Build record
    record = BBRecord(
        bb_id=parsed.bb_id,
        bb_type=BBType(parsed.bb_type),
        layer1_facts=layer1,
        layer2_semantics=layer2,
        provenance=provenance,
        validation_report=None,  # placeholder
    )

    # Enrich Layer 2 with SMARTS-based rules
    enrich_layer2(record)

    # Enrich node-specific fields (sbu_type, ligand_chemistry)
    enrich_node_fields(record)

    record.provenance.layer2_method = "smarts_rules"

    # Validate
    record.validation_report = validate_record(record)

    return record


def run_pipeline(
    bbs_dir: Path = BBS_DIR,
    output_dir: Path = OUTPUT_DIR,
    bb_ids: list[str] | None = None,
) -> dict:
    """Run the full pipeline on all (or selected) building blocks.

    Args:
        bbs_dir: Directory containing .xyz files.
        output_dir: Directory for output JSON files.
        bb_ids: Optional list of specific BB IDs to process (e.g., ["E1", "N100"]).

    Returns:
        Summary dict with counts and any errors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which files to process
    if bb_ids:
        files = []
        for bb_id in bb_ids:
            f = bbs_dir / f"{bb_id}.xyz"
            if f.exists():
                files.append(f)
            else:
                print(f"WARNING: {f} not found, skipping")
        files.sort(key=lambda f: f.stem)
    else:
        files = sorted(bbs_dir.glob("*.xyz"), key=lambda f: f.stem)

    total = len(files)
    results = {
        "total": total,
        "success": 0,
        "warnings": 0,
        "errors": 0,
        "parse_errors": [],
        "validation_errors": [],
    }

    print(f"Processing {total} building blocks...")

    for i, filepath in enumerate(files):
        bb_id = filepath.stem
        try:
            record = process_single(filepath)

            # Write JSON
            out_path = output_dir / f"{bb_id}.json"
            out_path.write_text(record.to_json(indent=2), encoding="utf-8")

            # Track status
            status = record.validation_report.status.value
            if status == "pass":
                results["success"] += 1
            elif status == "warn":
                results["warnings"] += 1
            elif status == "fail":
                results["errors"] += 1
                results["validation_errors"].append(
                    (bb_id, record.validation_report.errors)
                )

            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] {bb_id}: {status}")

        except XYZParseError as e:
            results["parse_errors"].append((bb_id, str(e)))
            print(f"  [{i+1}/{total}] {bb_id}: PARSE ERROR - {e}")
        except Exception as e:
            results["parse_errors"].append((bb_id, str(e)))
            print(f"  [{i+1}/{total}] {bb_id}: ERROR - {e}")

    # Summary
    print(f"\n=== Pipeline Complete ===")
    print(f"  Total:    {results['total']}")
    print(f"  Pass:     {results['success']}")
    print(f"  Warnings: {results['warnings']}")
    print(f"  Errors:   {results['errors']}")
    print(f"  Parse errors: {len(results['parse_errors'])}")
    print(f"  Output:   {output_dir}")

    # Write summary report
    summary_path = output_dir / "_pipeline_report.json"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "total_processed": results["total"],
        "pass_count": results["success"],
        "warn_count": results["warnings"],
        "error_count": results["errors"],
        "parse_error_count": len(results["parse_errors"]),
        "parse_errors": [
            {"bb_id": bb_id, "error": err}
            for bb_id, err in results["parse_errors"]
        ],
        "validation_errors": [
            {"bb_id": bb_id, "errors": errs}
            for bb_id, errs in results["validation_errors"]
        ],
    }
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PORMAKE BB metadata pipeline")
    parser.add_argument(
        "bb_ids", nargs="*", help="Specific BB IDs to process (default: all)"
    )
    parser.add_argument(
        "--bbs-dir", type=Path, default=BBS_DIR, help="Directory with .xyz files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory"
    )
    args = parser.parse_args()

    run_pipeline(
        bbs_dir=args.bbs_dir,
        output_dir=args.output_dir,
        bb_ids=args.bb_ids or None,
    )
