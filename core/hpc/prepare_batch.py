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
    geometry_filter: Dict[str, Any] = None,
) -> str:
    """
    Generate batch_manifest.json from ranked candidate pools.

    Args:
        beam_pools: {beam_id: [RankedMOF, ...]} — ranked candidates per beam
        iteration: Current iteration number
        experiment_id: Experiment directory name
        iter_dir: Path to iteration directory
        geometry_filter: Agent 2 geometry constraints for Beam Z pre-filtering.
            If provided and --use-zeo is active, Beam Z jobs skip RASPA when
            zeo++ geometry does not satisfy these constraints.

    Returns:
        Path to the generated manifest file
    """
    jobs = []
    job_idx = 0
    seen_filenames: set = set()

    for beam_id, ranked_mofs in beam_pools.items():
        for ranked_mof in ranked_mofs:
            comp = ranked_mof.component
            if comp.filename in seen_filenames:
                print(f"[Manifest] Skipping duplicate filename: {comp.filename} (beam {beam_id})")
                continue
            seen_filenames.add(comp.filename)
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
            "geometry_filter": geometry_filter or {},
        },
        "jobs": jobs,
    }

    manifest_path = os.path.join(iter_dir, "batch_manifest.json")
    os.makedirs(iter_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[Manifest] Written {len(jobs)} jobs to {manifest_path}")
    return manifest_path


def prepare_r1_manifest(
    z_beam_pool: List[Any],
    iteration: int,
    experiment_id: str,
    iter_dir: str,
    af_total_beam_pools: Dict[str, List[Any]] = None,
) -> str:
    """
    Generate batch_manifest_r1.json for Round-1: all beams stage1_only (pormake+lammps+zeo).

    Args:
        z_beam_pool: [RankedMOF, ...] — Z beam ranked candidates (up to LIVE_SIM_Z_POOL_SIZE)
        iteration: Current iteration number
        experiment_id: Experiment directory name
        iter_dir: Path to iteration directory
        af_total_beam_pools: {beam_id: [RankedMOF, ...]} for A/F/total beams (up to LIVE_SIM_AF_POOL_SIZE each)

    Returns:
        Path to the generated manifest file
    """
    jobs = []
    seen_filenames: set = set()

    for ranked_mof in z_beam_pool:
        comp = ranked_mof.component
        if comp.filename in seen_filenames:
            print(f"[Manifest R1] Skipping duplicate: {comp.filename}")
            continue
        seen_filenames.add(comp.filename)
        pred = ranked_mof.predicted_geometry
        pred_dict = pred.to_dict() if pred else None

        jobs.append({
            "job_idx": len(jobs),
            "beam_id": "Z",
            "filename": comp.filename,
            "topology": comp.topology,
            "node": comp.node,
            "edge": comp.edge,
            "predicted_geometry": pred_dict,
            "match_score": ranked_mof.match_score,
            "pipeline": "stage1_only",
        })

    for beam_id, ranked_mofs in (af_total_beam_pools or {}).items():
        for ranked_mof in ranked_mofs:
            comp = ranked_mof.component
            if comp.filename in seen_filenames:
                print(f"[Manifest R1] Skipping duplicate {beam_id}: {comp.filename}")
                continue
            seen_filenames.add(comp.filename)
            pred = ranked_mof.predicted_geometry
            pred_dict = pred.to_dict() if pred else None

            jobs.append({
                "job_idx": len(jobs),
                "beam_id": beam_id,
                "filename": comp.filename,
                "topology": comp.topology,
                "node": comp.node,
                "edge": comp.edge,
                "predicted_geometry": pred_dict,
                "match_score": ranked_mof.match_score,
                "pipeline": "stage1_only",
            })

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
            "geometry_filter": {},
        },
        "jobs": jobs,
    }

    manifest_path = os.path.join(iter_dir, "batch_manifest_r1.json")
    os.makedirs(iter_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n_z = sum(1 for j in jobs if j["beam_id"] == "Z")
    n_aft = len(jobs) - n_z
    print(f"[Manifest R1] Written {len(jobs)} stage1_only jobs to {manifest_path} "
          f"(Z: {n_z}, A/F/total: {n_aft})")
    return manifest_path


def prepare_r2_manifest(
    z_stage2_jobs: List[Dict],
    af_total_stage2_jobs: Dict[str, List[Dict]],
    iteration: int,
    experiment_id: str,
    iter_dir: str,
    geometry_filter: Dict[str, Any] = None,
) -> str:
    """
    Generate batch_manifest.json for Round-2: Z-top15 stage2_only + A/F/total stage2_only.

    Args:
        z_stage2_jobs: List of dicts from R1 Z filter (with cif_path, real_geometry)
        af_total_stage2_jobs: {beam_id: [dict, ...]} — top-N successes from R1 per beam
            Each dict: {filename, topology, node, edge, cif_path, real_geometry,
                        match_score, predicted_geometry}
        iteration: Current iteration number
        experiment_id: Experiment directory name
        iter_dir: Path to iteration directory
        geometry_filter: Agent 2 geometry constraints for manifest config

    Returns:
        Path to the generated manifest file
    """
    jobs = []
    seen_filenames: set = set()

    # Z stage2_only jobs (RASPA only, CIF from stage1)
    for j in z_stage2_jobs:
        if j["filename"] in seen_filenames:
            print(f"[Manifest R2] Skipping duplicate Z job: {j['filename']}")
            continue
        seen_filenames.add(j["filename"])
        jobs.append({
            "job_idx": len(jobs),
            "beam_id": "Z",
            "filename": j["filename"],
            "topology": j["topology"],
            "node": j["node"],
            "edge": j["edge"],
            "predicted_geometry": j.get("predicted_geometry"),
            "match_score": j.get("match_score", 0.0),
            "pipeline": "stage2_only",
            "cif_path": j["cif_path"],
            "real_geometry": j.get("real_geometry"),
        })

    # A/F/total stage2_only jobs (RASPA only, CIF from stage1)
    for beam_id, stage2_jobs in af_total_stage2_jobs.items():
        for j in stage2_jobs:
            if j["filename"] in seen_filenames:
                print(f"[Manifest R2] Skipping duplicate {beam_id} job: {j['filename']}")
                continue
            seen_filenames.add(j["filename"])
            jobs.append({
                "job_idx": len(jobs),
                "beam_id": beam_id,
                "filename": j["filename"],
                "topology": j["topology"],
                "node": j["node"],
                "edge": j["edge"],
                "predicted_geometry": j.get("predicted_geometry"),
                "match_score": j.get("match_score", 0.0),
                "pipeline": "stage2_only",
                "cif_path": j["cif_path"],
                "real_geometry": j.get("real_geometry"),
            })

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
            "geometry_filter": geometry_filter or {},
        },
        "jobs": jobs,
    }

    manifest_path = os.path.join(iter_dir, "batch_manifest.json")
    os.makedirs(iter_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n_z = sum(1 for j in jobs if j["beam_id"] == "Z")
    n_aft = len(jobs) - n_z
    print(f"[Manifest R2] Written {len(jobs)} jobs to {manifest_path} "
          f"(Z-stage2: {n_z}, A/F/total-stage2: {n_aft})")
    return manifest_path
