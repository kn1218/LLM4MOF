"""
Feedback Live Adapter — bridges LiveResults to FeedbackGenerator's expected format.

The existing feedback_generator._generate_four_beam() expects a dict of DataFrames
keyed by filter-set names ('z', 'a', 'f', 'total', etc.).  Each DataFrame must have:
  - 'filename' column  (e.g., "pcu+N1+E1")
  - 'target'  column   (the metric value in the user's requested unit)
  - Geometry columns: di, df, sa, vf, density, dif, cv  (from mof2zeo predictions)

This adapter converts live_runner.LiveResults into that format so the
feedback generator doesn't need any modifications.

Target column mapping by adsorbate:

  H2 (77 K, 100 bar):
    RASPA3 outputs: loading_mol_kg, loading_g_L, loading_mg_g
    target is unit-converted based on config.get_active_unit():
      - cm³(STP)/cm³ (default): loading_mol_kg × density × 22.414
      - g/L:                    loading_g_L (from RASPA3 or live_runner override)
      - mol/kg:                 loading_mol_kg

  CH4 (298 K, 2.5 bar):
    Same single-component parse path as H2.
    target units: same cm³(STP)/cm³ / g/L / mol/kg conversion, with MW=16.043 g/mol.

  CO2 (298 K, 2.5 bar):
    Same single-component parse path as H2/CH4.
    target units: same conversion, with MW=44.010 g/mol.
    NOTE: framework charges not yet assigned (assign_framework_charges is a stub);
    results are approximate until DDEC/EQeq charges are implemented.

  Xe/Kr mixture (273 K, 1 bar, 20% Xe / 80% Kr):
    RASPA3 outputs (parse_output_mixture): xe_loading_mol_kg, kr_loading_mol_kg,
      selectivity_xe_kr, xe_loading_g_L, kr_loading_g_L
    target = selectivity_xe_kr  (dimensionless; bypasses unit-conversion logic)
    diagnostic columns: xe_loading_mol_kg, kr_loading_mol_kg
"""

import pandas as pd
from typing import Dict

import config
from core.live_runner import LiveResults, SimResult


def _mol_kg_to_volumetric(loading_mol_kg: float, density_g_cm3: float) -> float:
    """Convert gravimetric uptake (mol/kg) to volumetric (cm³(STP)/cm³).

    cm³(STP)/cm³ = (mol/kg) × (1 kg / 1000 g) × (density g/cm³) × (22414 cm³(STP)/mol)
                 = mol/kg × density × 22.414
    """
    return loading_mol_kg * density_g_cm3 * config.MOLAR_VOL_STP_CM3_PER_MMOL


def _resolve_target(loading_mol_kg: float, loading_g_L: float,
                    density: float, active_unit: str) -> tuple[float | None, bool]:
    """Resolve the target value based on the user's active unit.

    Returns (target_value, density_was_missing).
    When density is missing and volumetric conversion is required:
      - STRICT_DENSITY_CHECK=True  → returns (None, True) to exclude the row
      - STRICT_DENSITY_CHECK=False → returns (loading_mol_kg, True) (legacy)
    """
    if active_unit == "g/L":
        return loading_g_L, False
    if active_unit == "mol/kg":
        return loading_mol_kg, False
    # Default: cm³(STP)/cm³ (volumetric) — requires density for conversion
    if density > 0:
        return _mol_kg_to_volumetric(loading_mol_kg, density), False
    # Density missing — cannot convert to volumetric
    if config.STRICT_DENSITY_CHECK:
        return None, True
    return loading_mol_kg, True


def _sim_results_to_dataframe(results: list[SimResult]) -> pd.DataFrame:
    """
    Convert a list of successful SimResults into a DataFrame matching
    the feedback generator's expected schema.

    For H2/CH4/CO2: 'target' is unit-converted loading (config.get_active_unit()).
    For Xe/Kr: 'target' is selectivity_xe_kr (dimensionless; no unit conversion).
    All RASPA3 native outputs are preserved as diagnostic columns.
    """
    if not results:
        return pd.DataFrame()

    active_unit = config.get_active_unit()
    is_xekr = getattr(config, "LIVE_SIM_ADSORBATE", "h2") == "xekr"
    rows = []
    n_density_missing = 0
    for r in results:
        if r.status != "success":
            continue

        uptake = r.real_uptake or {}
        # Use zeo++ computed geometry if available, fall back to PORMAKE prediction
        pred = r.real_geometry or r.predicted_geometry or {}

        if is_xekr:
            target = float(uptake.get("selectivity_xe_kr", 0.0))
            row = {
                "filename": r.filename,
                "target": target,
                "di": pred.get("di", 0.0),
                "df": pred.get("df", 0.0),
                "sa": pred.get("sa", 0.0),
                "vf": pred.get("vf", 0.0),
                "density": pred.get("density", 0.0),
                "dif": pred.get("dif", 0.0),
                "cv": pred.get("cv", 0.0),
                # Diagnostic columns
                "xe_loading_mol_kg": float(uptake.get("xe_loading_mol_kg", 0.0)),
                "kr_loading_mol_kg": float(uptake.get("kr_loading_mol_kg", 0.0)),
                "match_score": r.match_score,
                "geo_filter_passed": getattr(r, "geo_filter_passed", True),
                "geo_filter_fail_reason": getattr(r, "geo_filter_fail_reason", ""),
            }
        else:
            loading_mol_kg = float(uptake.get("loading_mol_kg", 0.0))
            loading_g_L = float(uptake.get("loading_g_L", 0.0))
            density = float(pred.get("density", 0.0))

            target, density_missing = _resolve_target(
                loading_mol_kg, loading_g_L, density, active_unit
            )
            if density_missing:
                n_density_missing += 1
                if target is None:
                    continue  # exclude row — cannot convert to volumetric

            row = {
                "filename": r.filename,
                "target": target,
                "di": pred.get("di", 0.0),
                "df": pred.get("df", 0.0),
                "sa": pred.get("sa", 0.0),
                "vf": pred.get("vf", 0.0),
                "density": pred.get("density", 0.0),
                "dif": pred.get("dif", 0.0),
                "cv": pred.get("cv", 0.0),
                # Diagnostic columns: all RASPA3 native outputs preserved
                "loading_mol_kg": loading_mol_kg,
                "loading_g_L": loading_g_L,
                "loading_mg_g": uptake.get("loading_mg_g", 0.0),
                "match_score": r.match_score,
                "geo_filter_passed": getattr(r, "geo_filter_passed", True),
                "geo_filter_fail_reason": getattr(r, "geo_filter_fail_reason", ""),
            }
        rows.append(row)

    if n_density_missing > 0:
        if config.STRICT_DENSITY_CHECK:
            print(f"[LiveAdapter] WARNING: {n_density_missing} MOFs excluded — "
                  f"missing density, cannot convert to {active_unit}")
        else:
            print(f"[LiveAdapter] WARNING: {n_density_missing} MOFs missing predicted density — "
                  f"target kept as mol/kg (not converted to {active_unit})")

    return pd.DataFrame(rows)


def live_results_to_filter_sets(live_results: LiveResults) -> Dict[str, pd.DataFrame]:
    """
    Convert LiveResults to the dict-of-DataFrames format the existing
    FeedbackGenerator.generate_feedback() expects.

    Mapping:
      Beam Z → filter_sets['z']     (full hypothesis: chemistry + geometry)
      Beam A → filter_sets['a']     (chemistry only)
      Beam F → filter_sets['f']     (metal only)
      Beam total → filter_sets['total']  (random baseline)

    Unused filter sets (d, e, e2, g) are set to empty DataFrames.
    The feedback generator handles empty sets gracefully.
    """
    beams = live_results.beams

    z_beam = beams.get("Z")
    a_beam = beams.get("A")
    f_beam = beams.get("F")
    total_beam = beams.get("total")

    filter_sets = {
        "z": _sim_results_to_dataframe(z_beam.successes if z_beam else []),
        "a": _sim_results_to_dataframe(a_beam.successes if a_beam else []),
        "f": _sim_results_to_dataframe(f_beam.successes if f_beam else []),
        "total": _sim_results_to_dataframe(total_beam.successes if total_beam else []),
        # Unused in live mode — set to empty
        "d": pd.DataFrame(),
        "e": pd.DataFrame(),
        "e2": pd.DataFrame(),
        "g": pd.DataFrame(),
    }

    # Include per-beam matchmaker diagnostics for use in diagnostic footer
    filter_sets["_diag_info"] = {
        "Z": z_beam.matchmaker_diag if z_beam else {},
        "A": a_beam.matchmaker_diag if a_beam else {},
        "F": f_beam.matchmaker_diag if f_beam else {},
    }


    # Log summary with unit info
    is_xekr = getattr(config, "LIVE_SIM_ADSORBATE", "h2") == "xekr"
    unit = "selectivity" if is_xekr else config.get_active_unit()
    for key, df in filter_sets.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            avg_target = df["target"].mean()
            extra = ""
            if is_xekr and "xe_loading_mol_kg" in df.columns:
                extra = f", avg Xe={df['xe_loading_mol_kg'].mean():.3f} mol/kg, avg Kr={df['kr_loading_mol_kg'].mean():.3f} mol/kg"
            elif "loading_mol_kg" in df.columns:
                extra = f", avg mol/kg={df['loading_mol_kg'].mean():.2f}"
            print(f"[LiveAdapter] Set '{key}': {len(df)} entries, "
                  f"avg target={avg_target:.2f} {unit}{extra}")

    return filter_sets
