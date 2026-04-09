"""
Prepare Batch Manifest — generates batch_manifest.json from ranked candidate pools.

Called during --prepare phase of run_live_experiment.py.
Produces a JSON file that is uploaded to HPC for array job execution.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def prepare_manifest(
    beam_pools: Dict[str, List[Any]],
    iteration: int,
    experiment_id: str,
    iter_dir: str,
) -> str:
    """
    Generate batch_manifest.json from ranked candidate pools.

    Args:
        beam_pools: {beam_id: [RankedMOF, ...]} — ranked candidates per beam
        iteration: Current iteration number
        experiment_id: Experiment directory name
        iter_dir: Path to iteration directory

    Returns:
        Path to the generated manifest file
    """
    jobs = []
    job_idx = 0

    for beam_id, ranked_mofs in beam_pools.items():
        for ranked_mof in ranked_mofs:
            comp = ranked_mof.component
            pred = ranked_mof.predicted_geometry
            pred_dict = pred.to_dict() if pred else None

            jobs.append({
                "job_idx": job_idx,
                "beam_id": beam_id,
                "filename": comp.filename,
                "topology": comp.topology,
                "node": comp.node,
                "edge": comp.edge,
                "predicted_geometry": pred_dict,
                "match_score": ranked_mof.match_score,
            })
            job_idx += 1

    manifest = {
        "version": 1,
        "experiment_id": experiment_id,
        "iteration": iteration,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(jobs),
        "config": {
            "raspa_cycles": config.LIVE_SIM_RASPA_CYCLES,
            "raspa_init_cycles": config.LIVE_SIM_RASPA_INIT_CYCLES,
            "temperature": config.LIVE_SIM_RASPA_TEMPERATURE,
            "pressure": config.LIVE_SIM_RASPA_PRESSURE,
            "skip_lammps": config.LIVE_SIM_SKIP_LAMMPS,
            "lammps_timeout": config.LIVE_SIM_LAMMPS_TIMEOUT,
            "raspa_timeout": config.LIVE_SIM_RASPA_TIMEOUT,
            "helium_void_fraction": 0.5,
        },
        "jobs": jobs,
    }

    manifest_path = os.path.join(iter_dir, "batch_manifest.json")
    os.makedirs(iter_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[Manifest] Written {len(jobs)} jobs to {manifest_path}")
    return manifest_path
