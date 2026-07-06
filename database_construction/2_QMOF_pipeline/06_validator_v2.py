"""Phase 6: QMOF-adapted validation layer with 14 automated checks.

Each check maps to a specific quality dimension. Checks are categorized
by severity: some failures produce errors (data integrity), others
produce warnings (data completeness).

Error-severity checks (data unusable if failed):
  - formula_nonempty, bandgap_nonneg, density_positive, volume_positive

Warning-severity checks (data incomplete but usable):
  - has_smiles_nodes, has_smiles_linkers, smiles_nodes_all_valid,
    smiles_linkers_all_valid, has_metal_node_data,
    metal_composition_consistent, topology_present, fg_detection_ran,
    source_json_available, provenance_complete
"""

from __future__ import annotations

from .schema_v2 import QMOFRecordV2, ValidationReport, IssueLogEntry


def validate_record(record: QMOFRecordV2) -> ValidationReport:
    """Run all 14 validation checks on a QMOFRecordV2 record.

    Populates record.validation_report with check results and computes
    overall status. Does NOT modify any data fields — read-only analysis.

    Args:
        record: Fully populated QMOFRecordV2 record.

    Returns:
        ValidationReport with all checks, warnings, errors, and status.
    """
    l1 = record.layer1_facts
    l2 = record.layer2_semantics
    prov = record.provenance

    checks: dict[str, bool] = {}
    warnings: list[str] = []
    errors: list[str] = []

    # ── Error-severity checks ──────────────────────────────────────────

    # 1. Formula non-empty
    checks["formula_nonempty"] = bool(l1.formula and l1.formula.strip())
    if not checks["formula_nonempty"]:
        errors.append("formula is empty or missing")

    # 9. Bandgap non-negative (only check if present)
    bandgaps = [
        ("bandgap_pbe", l1.bandgap_pbe),
        ("bandgap_hle17", l1.bandgap_hle17),
        ("bandgap_hse06_10hf", l1.bandgap_hse06_10hf),
        ("bandgap_hse06", l1.bandgap_hse06),
    ]
    bandgap_ok = True
    for name, val in bandgaps:
        if val is not None and val < 0:
            bandgap_ok = False
            errors.append(f"{name} is negative: {val}")
    checks["bandgap_nonneg"] = bandgap_ok

    # 10. Density positive (only check if present)
    if l1.density is not None:
        checks["density_positive"] = l1.density > 0
        if not checks["density_positive"]:
            errors.append(f"density is non-positive: {l1.density}")
    else:
        checks["density_positive"] = True  # missing is OK, not an error

    # 11. Volume positive (only check if present)
    if l1.volume is not None:
        checks["volume_positive"] = l1.volume > 0
        if not checks["volume_positive"]:
            errors.append(f"volume is non-positive: {l1.volume}")
    else:
        checks["volume_positive"] = True  # missing is OK

    # ── Warning-severity checks ────────────────────────────────────────

    # 2. Has SMILES nodes
    checks["has_smiles_nodes"] = bool(l1.smiles_nodes)
    if not checks["has_smiles_nodes"]:
        warnings.append("no SMILES nodes available")

    # 3. Has SMILES linkers
    checks["has_smiles_linkers"] = bool(l1.smiles_linkers)
    if not checks["has_smiles_linkers"]:
        warnings.append("no SMILES linkers available")

    # 4. SMILES nodes all valid
    if l1.smiles_nodes_valid:
        checks["smiles_nodes_all_valid"] = all(l1.smiles_nodes_valid)
        if not checks["smiles_nodes_all_valid"]:
            n_invalid = sum(1 for v in l1.smiles_nodes_valid if not v)
            warnings.append(f"{n_invalid}/{len(l1.smiles_nodes_valid)} node SMILES invalid")
    else:
        checks["smiles_nodes_all_valid"] = not bool(l1.smiles_nodes)

    # 5. SMILES linkers all valid
    if l1.smiles_linkers_valid:
        checks["smiles_linkers_all_valid"] = all(l1.smiles_linkers_valid)
        if not checks["smiles_linkers_all_valid"]:
            n_invalid = sum(1 for v in l1.smiles_linkers_valid if not v)
            warnings.append(f"{n_invalid}/{len(l1.smiles_linkers_valid)} linker SMILES invalid")
    else:
        checks["smiles_linkers_all_valid"] = not bool(l1.smiles_linkers)

    # 6. Has metal node data
    mn = l1.metal_node
    checks["has_metal_node_data"] = bool(mn.metals)
    if not checks["has_metal_node_data"]:
        warnings.append("no metal node data (metals list empty)")

    # 7. Metal composition consistent (formula vs source JSON)
    # This is checked during enrichment and logged in issue_log.
    # Here we just check if any cross-validation issues were logged.
    metal_issues = [
        i for i in record.validation_report.issue_log
        if i.field == "metal_composition" and i.source == "cross_validation"
    ]
    checks["metal_composition_consistent"] = len(metal_issues) == 0
    if not checks["metal_composition_consistent"]:
        warnings.append(f"metal composition inconsistency found ({len(metal_issues)} issues)")

    # 8. Topology present
    checks["topology_present"] = bool(l1.topology and l1.topology.strip())
    if not checks["topology_present"]:
        warnings.append("no topology information")

    # 12. FG detection ran (if SMILES available, FGs should be detected)
    has_valid_smiles = any(l1.smiles_linkers_valid) if l1.smiles_linkers_valid else False
    if has_valid_smiles:
        checks["fg_detection_ran"] = bool(l2.functional_groups.rule_based)
        if not checks["fg_detection_ran"]:
            warnings.append("valid linker SMILES present but no FGs detected")
    else:
        checks["fg_detection_ran"] = True  # no valid SMILES = no FG detection expected

    # 13. Source JSON available
    checks["source_json_available"] = bool(prov.source_json)
    if not checks["source_json_available"]:
        warnings.append("source analysis JSON not found")

    # 14. Provenance complete
    prov_complete = bool(
        prov.generated_at
        and prov.tool_versions
        and prov.pipeline_version
        and prov.schema_version
    )
    checks["provenance_complete"] = prov_complete
    if not prov_complete:
        warnings.append("provenance metadata incomplete")

    # ── Compute status ─────────────────────────────────────────────────

    report = ValidationReport(
        checks=checks,
        warnings=warnings,
        errors=errors,
        issue_log=record.validation_report.issue_log,  # preserve existing issues
    )
    report.compute_status()

    return report


def validate_batch(records: list[QMOFRecordV2]) -> dict:
    """Validate a batch of records and return aggregate statistics.

    Returns dict with:
      - total, pass_count, warning_count, error_count
      - check_stats: per-check pass rate
      - top_warnings: most common warnings
      - top_errors: most common errors
    """
    from collections import Counter

    total = len(records)
    pass_count = 0
    warning_count = 0
    error_count = 0
    check_pass_counts: dict[str, int] = {}
    all_warnings: list[str] = []
    all_errors: list[str] = []

    for record in records:
        report = validate_record(record)
        record.validation_report = report

        if report.status == "pass":
            pass_count += 1
        elif report.status == "warning":
            warning_count += 1
        else:
            error_count += 1

        for check_name, passed in report.checks.items():
            if check_name not in check_pass_counts:
                check_pass_counts[check_name] = 0
            if passed:
                check_pass_counts[check_name] += 1

        all_warnings.extend(report.warnings)
        all_errors.extend(report.errors)

    # Compute per-check pass rate
    check_stats = {
        name: {"pass_count": count, "pass_pct": round(100 * count / total, 2) if total else 0}
        for name, count in check_pass_counts.items()
    }

    # Top warnings/errors
    warning_counter = Counter(all_warnings)
    error_counter = Counter(all_errors)

    return {
        "total": total,
        "pass_count": pass_count,
        "warning_count": warning_count,
        "error_count": error_count,
        "check_stats": check_stats,
        "top_warnings": warning_counter.most_common(10),
        "top_errors": error_counter.most_common(10),
    }
