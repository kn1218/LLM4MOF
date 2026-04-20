"""
Feedback Live Adapter — bridges LiveResults to FeedbackGenerator's expected format.

The existing feedback_generator._generate_four_beam() expects a dict of DataFrames
keyed by filter-set names ('z', 'a', 'f', 'total', etc.).  Each DataFrame must have:
  - 'filename' column  (e.g., "pcu+N1+E1")
  - 'target'  column   (the metric value in the user's requested unit)
  - Geometry columns: di, df, sa, vf, density, dif, cv  (from mof2zeo predictions)

This adapter converts live_runner.LiveResults into that format so the
feedback generator doesn't need any modifications.

Unit handling (2026-04-14):
  RASPA3 outputs H2 uptake in three units natively:
    - loading_mol_kg  (mol/kg, gravimetric)
    - loading_g_L     (g/L, volumetric mass-concentration)
    - loading_mg_g    (mg/g, diagnostic only)

  The 'target' column is set based on the user's active unit:
    - cm³(STP)/cm³ (default): mol/kg × density × 22.414
    - g/L:                    loading_g_L from RASPA3 directly
    - mol/kg:                 loading_mol_kg from RASPA3 directly
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
                    density: float, active_unit: str) -> tuple[float, bool]:
    """Resolve the target value based on the user's active unit.

    Returns (target_value, density_was_missing).
    """
    if active_unit == "g/L":
        return loading_g_L, False
    if active_unit == "mol/kg":
        return loading_mol_kg, False
    # Default: cm³(STP)/cm³ (volumetric) — requires density for conversion
    if density > 0:
        return _mol_kg_to_volumetric(loading_mol_kg, density), False
    return loading_mol_kg, True


def _sim_results_to_dataframe(results: list[SimResult]) -> pd.DataFrame:
    """
    Convert a list of successful SimResults into a DataFrame matching
    the feedback generator's expected schema.

    The 'target' column is set in the user's requested unit (determined by
    config.get_active_unit()). All three RASPA3 native outputs are preserved
    as diagnostic columns.
    """
    if not results:
        return pd.DataFrame()

    active_unit = config.get_active_unit()
    rows = []
    n_density_missing = 0
    for r in results:
        if r.status != "success":
            continue

        uptake = r.real_uptake or {}
        pred = r.predicted_geometry or {}

        loading_mol_kg = float(uptake.get("loading_mol_kg", 0.0))
        loading_g_L = float(uptake.get("loading_g_L", 0.0))
        density = float(pred.get("density", 0.0))

        target, density_missing = _resolve_target(
            loading_mol_kg, loading_g_L, density, active_unit
        )
        if density_missing:
            n_density_missing += 1

        rows.append({
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
        })

    if n_density_missing > 0:
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
    unit = config.get_active_unit()
    for key, df in filter_sets.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            avg_target = df["target"].mean()
            extra = ""
            if "loading_mol_kg" in df.columns:
                avg_mol_kg = df["loading_mol_kg"].mean()
                extra = f", avg mol/kg={avg_mol_kg:.2f}"
            print(f"[LiveAdapter] Set '{key}': {len(df)} entries, "
                  f"avg target={avg_target:.2f} {unit}{extra}")

    return filter_sets
