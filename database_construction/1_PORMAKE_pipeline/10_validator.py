"""Phase 5: Self-validation for building block metadata.

Each BB record gets a ValidationReport with per-check results,
warnings, and errors. Checks are deterministic and reproducible.
"""

from __future__ import annotations

from .schema import (
    BBRecord,
    Layer1Facts,
    ValidationReport,
    ValidationStatus,
)
from .config import METAL_SET


def validate_record(record: BBRecord) -> ValidationReport:
    """Run all validation checks on a BBRecord.

    Returns:
        ValidationReport with status, check results, warnings, errors.
    """
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    errors: list[str] = []

    facts = record.layer1_facts

    # ── Check 1: Formula non-empty ──
    checks["formula_nonempty"] = bool(facts.formula)
    if not checks["formula_nonempty"]:
        errors.append("Formula is empty")

    # ── Check 2: Molecular weight positive ──
    checks["mw_positive"] = facts.molecular_weight > 0
    if not checks["mw_positive"]:
        errors.append("Molecular weight is zero or negative")

    # ── Check 3: Total atoms > 0 ──
    checks["atoms_positive"] = facts.total_atoms > 0
    if not checks["atoms_positive"]:
        errors.append("No real atoms found")

    # ── Check 4: Atom count sum matches total ──
    atom_sum = sum(facts.atom_counts.values())
    checks["atom_count_consistent"] = atom_sum == facts.total_atoms
    if not checks["atom_count_consistent"]:
        errors.append(
            f"Atom count sum ({atom_sum}) != total_atoms ({facts.total_atoms})"
        )

    # ── Check 5: Connection points > 0 ──
    checks["has_connection_points"] = facts.connection_points.count > 0
    if not checks["has_connection_points"]:
        errors.append("No connection points (dummy atoms) found")

    # ── Check 6: Connection point count matches points list ──
    checks["cp_count_consistent"] = (
        facts.connection_points.count == len(facts.connection_points.points)
    )
    if not checks["cp_count_consistent"]:
        errors.append("Connection point count mismatch")

    # ── Check 7: Bond graph has bonds ──
    checks["has_bonds"] = facts.bond_graph.num_bonds > 0
    if not checks["has_bonds"]:
        warnings.append("No bonds in real atom graph")

    # ── Check 8: Bond graph connectivity ──
    checks["graph_connected"] = facts.bond_graph.is_connected
    if not checks["graph_connected"]:
        warnings.append("Real atom graph is disconnected")

    # ── Check 9: SMILES generated ──
    checks["smiles_generated"] = facts.smiles is not None
    if not checks["smiles_generated"]:
        warnings.append(f"SMILES generation failed (method: {facts.smiles_method})")

    # ── Check 10: SMILES contains connection points ──
    if facts.smiles:
        checks["smiles_has_wildcard"] = "[*]" in facts.smiles
        if not checks["smiles_has_wildcard"]:
            warnings.append("SMILES does not contain [*] (connection points)")

    # ── Check 11: DoU non-negative for organic BBs ──
    if not facts.has_metal:
        checks["dou_nonneg"] = facts.degree_of_unsaturation >= 0
        if not checks["dou_nonneg"]:
            warnings.append(
                f"Negative DoU ({facts.degree_of_unsaturation}) for organic BB"
            )

    # ── Check 12: Edge-specific checks ──
    if record.bb_type.value == "edge":
        checks["edge_has_2cp"] = facts.connection_points.count >= 2
        if not checks["edge_has_2cp"]:
            warnings.append(
                f"Edge has {facts.connection_points.count} connection points (expected >= 2)"
            )
        if facts.length_angstroms is not None:
            checks["edge_length_positive"] = facts.length_angstroms > 0
            if not checks["edge_length_positive"]:
                warnings.append("Edge length is zero or negative")

    # ── Check 13: Node-specific checks ──
    if record.bb_type.value == "node":
        if facts.nuclearity is not None:
            checks["node_nuclearity_nonneg"] = facts.nuclearity >= 0
        if facts.has_metal and facts.metal_coordination:
            for mc in facts.metal_coordination:
                if mc.coordination_number == 0:
                    warnings.append(
                        f"Metal {mc.element} at index {mc.metal_index} has 0 bonds"
                    )

    # ── Check 14: Metals list consistency ──
    checks["metals_consistent"] = facts.has_metal == (len(facts.metals) > 0)
    if not checks["metals_consistent"]:
        errors.append("has_metal flag inconsistent with metals list")

    # ── Determine overall status ──
    if errors:
        status = ValidationStatus.FAIL
    elif warnings:
        status = ValidationStatus.WARN
    else:
        status = ValidationStatus.PASS

    return ValidationReport(
        status=status,
        checks=checks,
        warnings=warnings,
        errors=errors,
    )
