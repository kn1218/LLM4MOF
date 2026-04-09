"""
Collect Results — parses batch_results.json into LiveResults.

Called during --collect phase of run_live_experiment.py.
Reads the aggregated results downloaded from HPC and converts
them into the LiveResults format the feedback generator expects.
"""

import json
import os
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from core.live_runner import SimResult, BeamResult, LiveResults, SimCache


def collect_results(
    results_path: str,
    sim_cache: SimCache,
    n_per_beam: int = config.LIVE_SIM_N_PER_BEAM,
) -> LiveResults:
    """
    Parse batch_results.json into LiveResults.

    Args:
        results_path: Path to batch_results.json (downloaded from HPC)
        sim_cache: Persistent simulation cache to update
        n_per_beam: Target successes per beam

    Returns:
        LiveResults with all beam outcomes
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    live_results = LiveResults()

    # Group results by beam
    beam_results_map: dict[str, list[dict]] = {}
    for result in data.get("results", []):
        beam_id = result.get("beam_id", "unknown")
        beam_results_map.setdefault(beam_id, []).append(result)

    beam_labels = {
        "Z": "Full Hypothesis (Chemistry + Geometry)",
        "A": "Chemistry Only (no geometry gate)",
        "F": "Metal Only (no linker constraints)",
        "total": "Random Baseline (global)",
    }

    for beam_id in ["Z", "A", "F", "total"]:
        results_list = beam_results_map.get(beam_id, [])
        beam_label = beam_labels.get(beam_id, beam_id)

        beam_result = BeamResult(
            beam_id=beam_id,
            beam_label=beam_label,
            pool_size=len(results_list),
            target_n=n_per_beam,
        )

        for r in results_list:
            sim_result = SimResult(
                topology=r.get("topology", ""),
                node=r.get("node", ""),
                edge=r.get("edge", ""),
                filename=r.get("filename", ""),
                status=r.get("status", "unknown"),
                predicted_geometry=r.get("predicted_geometry"),
                match_score=r.get("match_score", 0.0),
                real_uptake=r.get("real_uptake"),
                error_msg=r.get("error_msg", ""),
                wall_seconds=r.get("wall_seconds", 0.0),
            )

            # Update cache
            sim_cache.put(sim_result)

            if sim_result.status == "success":
                beam_result.successes.append(sim_result)
                live_results.n_real_simulations += 1
            else:
                beam_result.failures.append(sim_result)
                live_results.n_failures += 1

        # Check beam status
        if not beam_result.is_complete and not beam_result.is_acceptable:
            if beam_result.pool_size > 0:
                live_results.aborted_beams.append(beam_id)

        live_results.beams[beam_id] = beam_result

        n_ok = len(beam_result.successes)
        n_fail = len(beam_result.failures)
        print(f"[Collect] Beam {beam_id}: {n_ok} successes, {n_fail} failures")

    live_results.wall_clock_seconds = data.get("total_wall_seconds", 0.0)

    print(f"[Collect] Total: {live_results.n_real_simulations} successes, "
          f"{live_results.n_failures} failures")

    return live_results
