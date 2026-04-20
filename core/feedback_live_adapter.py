"""
Feedback Live Adapter — bridges LiveResults to FeedbackGenerator's expected format.

The existing feedback_generator._generate_four_beam() expects a dict of DataFrames
keyed by filter-set names ('z', 'a', 'f', 'total', etc.).  Each DataFrame must have:
  - 'filename' column  (e.g., "pcu+N1+E1")
  - 'target'  column   (the metric value — H2 uptake in mol/kg)
  - Geometry columns: di, df, sa, vf, density, dif, cv  (from mof2zeo predictions)

This adapter converts live_runner.LiveResults into that format so the
feedback generator doesn't need any modifications.
"""

import pandas as pd
from typing import Dict

from core.live_runner import LiveResults, SimResult


def _sim_results_to_dataframe(results: list[SimResult]) -> pd.DataFrame:
    """
    Convert a list of successful SimResults into a DataFrame matching
    the feedback generator's expected schema.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        if r.status != "success":
            continue

        uptake = r.real_uptake or {}
        pred = r.predicted_geometry or {}

        rows.append({
            "filename": r.filename,
            "target": uptake.get("loading_mol_kg", 0.0),
            "di": pred.get("di", 0.0),
            "df": pred.get("df", 0.0),
            "sa": pred.get("sa", 0.0),
            "vf": pred.get("vf", 0.0),
            "density": pred.get("density", 0.0),
            "dif": pred.get("dif", 0.0),
            "cv": pred.get("cv", 0.0),
            # Extra columns for diagnostics (not consumed by feedback_generator
            # but useful for logging/analysis)
            "loading_g_L": uptake.get("loading_g_L", 0.0),
            "loading_mg_g": uptake.get("loading_mg_g", 0.0),
            "match_score": r.match_score,
        })

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

    # Log summary
    for key, df in filter_sets.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            avg_target = df["target"].mean()
            print(f"[LiveAdapter] Set '{key}': {len(df)} entries, "
                  f"avg target={avg_target:.2f}")

    return filter_sets
