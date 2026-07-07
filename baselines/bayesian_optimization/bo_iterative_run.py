"""
bo_iterative_run.py — Iterative LVGP-BO baseline (per-rep) for comparison with LLM4MOF.

Starting from iter1 data (~40 MOFs), runs 9 additional iterations:
  iter k (k=2..10):
    1. Train LVGP surrogate on accumulated data (85:15 train/val)
    2. Enumerate PORMAKE candidate space → score by EI → select top-80
    3. Simulate top-80 on HPC → poll to completion
    4. Take top-40 by EI rank (successful sims only) → add to training data

Run 5 reps in parallel (one process per rep):
    conda run -n <env> python bo_iterative_run.py --rep 1
    conda run -n <env> python bo_iterative_run.py --rep 2
    ...

Resume after interruption:
    conda run -n <env> python bo_iterative_run.py --rep 1 --start-iter 5
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GA_SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "test_260618_ga"))
PROJ_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.insert(0, SCRIPT_DIR)      # lvgp_surrogate.py lives here
sys.path.insert(0, GA_SRC_DIR)      # ga_hpc_worker, ga_topo
sys.path.insert(0, os.path.join(PROJ_ROOT, "core", "mof2zeo"))
sys.path.insert(0, os.path.join(PROJ_ROOT, "core"))
sys.path.insert(0, PROJ_ROOT)

import config
from lvgp_surrogate import (
    train_lvgp, save_checkpoint, load_checkpoint,
    predict_with_uncertainty, expected_improvement,
)
from ga_topo import load_pormake_dicts, build_topo_search_spaces

# ---------------------------------------------------------------------------
# HPC constants
# ---------------------------------------------------------------------------
HPC_HOST        = config.HPC_HOST
HPC_BASE        = "<HPC_WORK>/test_260619_bo"
HPC_GA_SRC      = "<HPC_WORK>/test_260618_ga"
HPC_PYTHON_SIM  = "python"
HPC_LAMMPS      = "/home/apps/opt/lammps/200303/bin"
NODE_PROP       = config.HPC_NODE_PROPERTY

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------
BO_MAX_CANDS_PER_TOPO = 100    # enumerate up to N (node,edge) pairs per topology for EI scoring
BO_TOP_N_PER_TOPO     = 3      # keep top-N by EI per topology → ~952×3=2,856 pool → global top-80
SIM_TOP_N             = 80     # simulate top-80 by EI
TRAIN_TOP_N           = 40     # add top-40 (by EI rank, sim-success only) to training data
LVGP_N_STEPS          = 300
LVGP_LR               = 0.05

POLL_INTERVAL  = 120  # seconds between polls
MAX_WAIT_HOURS = 6    # maximum wait per phase


# ===========================================================================
# SSH helpers  (verbatim from ga_iterative_run.py)
# ===========================================================================

def _ssh_run(cmd: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    full = f"export PATH=/usr/local/pbs/bin:/usr/local/mjs:{HPC_LAMMPS}:$PATH; {cmd}"
    delays = config.HPC_SSH_RETRY_DELAYS
    for attempt in range(config.HPC_SSH_RETRIES):
        try:
            r = subprocess.run(["ssh", HPC_HOST, full],
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0 or not check:
                return r
            if r.returncode != 255:
                return r
            print(f"   [SSH] attempt {attempt+1}: {r.stderr.strip()[:80]}")
        except subprocess.TimeoutExpired:
            print(f"   [SSH] timeout (attempt {attempt+1})")
        if attempt < config.HPC_SSH_RETRIES - 1:
            d = delays[min(attempt, len(delays) - 1)]
            print(f"   [SSH] retry in {d}s...")
            time.sleep(d)
    raise ConnectionError(f"SSH to {HPC_HOST} failed")


def _rsync_up(local: str, remote: str) -> None:
    subprocess.run(["rsync", "-avz", f"{local}/", f"{HPC_HOST}:{remote}/"], check=True)


def _rsync_down(remote: str, local: str, include_json: bool = True) -> None:
    os.makedirs(local, exist_ok=True)
    if include_json:
        subprocess.run(
            ["rsync", "-avz",
             "--include=*/", "--include=*.json", "--exclude=*",
             f"{HPC_HOST}:{remote}/", f"{local}/"],
            check=True,
        )
    else:
        subprocess.run(["rsync", "-avz", f"{HPC_HOST}:{remote}/", f"{local}/"], check=True)


# ===========================================================================
# Polling  (verbatim from ga_iterative_run.py)
# ===========================================================================

def _poll_jobs(
    hpc_results_dir: str,
    local_results_dir: str,
    n_jobs: int,
    job_prefix: str,
    qsub_dir: str,
    all_names: list[str],
    result_subpath: str = "result.json",
) -> int:
    max_polls = int(MAX_WAIT_HOURS * 3600 / POLL_INTERVAL)

    def _rpath(name: str) -> str:
        if result_subpath:
            return os.path.join(local_results_dir, name, result_subpath)
        return os.path.join(local_results_dir, name)

    for poll in range(1, max_polls + 1):
        time.sleep(POLL_INTERVAL)
        _rsync_down(hpc_results_dir, local_results_dir)

        done_names = set(n for n in all_names if os.path.isfile(_rpath(n)))
        n_done = len(done_names)

        q_result = _ssh_run(
            f"(myqstat 2>/dev/null; myqinfo 2>/dev/null) | grep {job_prefix} | wc -l",
            check=False,
        )
        try:
            queue_empty = int(q_result.stdout.strip()) == 0
        except ValueError:
            queue_empty = False

        elapsed = poll * POLL_INTERVAL / 60
        print(f"   [poll #{poll}] {n_done}/{n_jobs} done  queue_empty={queue_empty}"
              f"  ({elapsed:.0f} min)", flush=True)

        if n_done >= n_jobs:
            print(f"   [poll] All {n_jobs} jobs complete.")
            break

        if queue_empty:
            _rsync_down(hpc_results_dir, local_results_dir)
            done_names = set(n for n in all_names if os.path.isfile(_rpath(n)))
            n_done = len(done_names)
            print(f"   [poll] Queue empty. {n_done}/{n_jobs} done. Proceeding.", flush=True)
            break

    return n_done


# ===========================================================================
# Step 1 — LVGP training
# ===========================================================================

def step_train_lvgp(
    accumulated: pd.DataFrame,
    ckpt_path: str,
    rep: int,
    iter_k: int,
    local_iter_dir: str,
) -> str:
    """Train LVGP on accumulated data; save checkpoint + val metrics. Returns ckpt_path."""
    print(f"\n[iter{iter_k}] Training LVGP  n={len(accumulated)}", flush=True)

    data_list = []
    for _, row in accumulated.iterrows():
        parts = str(row["filename"]).split("+")
        if len(parts) != 3:
            continue
        data_list.append({
            "topo": parts[0], "node": parts[1], "edge": parts[2],
            "target": float(row["uptake"]),
        })

    if len(data_list) < 4:
        raise RuntimeError(f"Too few training samples: {len(data_list)}")

    model, likelihood, scaler, val_metrics = train_lvgp(
        data_list,
        n_steps=LVGP_N_STEPS,
        lr=LVGP_LR,
        seed=rep * 100 + iter_k,
        device="cpu",
    )
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    save_checkpoint(model, likelihood, scaler, ckpt_path)

    # Save val metrics
    metrics_path = os.path.join(local_iter_dir, "lvgp_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"rep": rep, "iter": iter_k, **val_metrics}, f, indent=2)

    # Append to global CSV
    global_csv = os.path.join(SCRIPT_DIR, "results", "lvgp_eval.csv")
    os.makedirs(os.path.dirname(global_csv), exist_ok=True)
    write_header = not os.path.isfile(global_csv)
    with open(global_csv, "a", newline="") as f:
        row = {"rep": rep, "iter": iter_k,
               "n_train": val_metrics.get("n_train", ""),
               "n_val": val_metrics.get("n_val", ""),
               "r2": round(val_metrics.get("val_r2", float("nan")), 2),
               "mae": round(val_metrics.get("val_mae", float("nan")), 2)}
        w = csv_module.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    r2  = val_metrics.get("val_r2",  "N/A")
    mae = val_metrics.get("val_mae", "N/A")
    print(f"[iter{iter_k}] LVGP saved → {ckpt_path}  val_r2={r2}  val_mae={mae}", flush=True)
    return ckpt_path


# ===========================================================================
# Step 2 — Candidate generation via EI
# ===========================================================================

def step_generate_candidates_bo(
    ckpt_path: str,
    spaces: dict,
    accumulated: pd.DataFrame,
    rep: int,
    iter_k: int,
    top_n: int = SIM_TOP_N,
) -> pd.DataFrame:
    """
    Enumerate PORMAKE space, score by EI.
    Strategy (Option A, mirrors GA):
      1. Per topology: enumerate up to BO_MAX_CANDS_PER_TOPO pairs, score by EI, keep top-3
      2. Global: from ~952×3 pool, select top_n by EI
    Excludes any MOF already in accumulated.
    """
    model, likelihood, scaler = load_checkpoint(ckpt_path, device="cpu")
    seen = set(accumulated["filename"].tolist())
    f_best = float(accumulated["uptake"].max())

    rng = random.Random(rep * 100 + iter_k)

    # Enumerate candidates per topology
    topo_cand_keys = {}   # topo_id → list of keys
    for topo_id, sp in spaces.items():
        node_list = sp["possible_node_bbs"][0]   # list of node IDs
        edge_list = sp["possible_edge_bbs"]       # list of edge IDs
        pairs_list = [(n, e) for n in node_list for e in edge_list]
        if len(pairs_list) > BO_MAX_CANDS_PER_TOPO:
            pairs_list = rng.sample(pairs_list, BO_MAX_CANDS_PER_TOPO)
        keys = [f"{topo_id}+{node}+{edge}" for node, edge in pairs_list
                if f"{topo_id}+{node}+{edge}" not in seen]
        if keys:
            topo_cand_keys[topo_id] = keys

    all_keys = [k for keys in topo_cand_keys.values() for k in keys]
    if not all_keys:
        return pd.DataFrame()

    print(f"[iter{iter_k}] BO candidates: {len(all_keys)} enumerated across"
          f" {len(topo_cand_keys)} topos  f_best={f_best:.3f}", flush=True)

    # Predict EI for all candidates
    mean, std, valid_idx = predict_with_uncertainty(
        model, likelihood, scaler, all_keys, device="cpu"
    )
    if len(valid_idx) == 0:
        return pd.DataFrame()

    valid_keys = [all_keys[i] for i in valid_idx]
    ei = expected_improvement(mean, std, f_best)

    # Build per-key lookup
    key_to_score = {k: (float(mean[i]), float(std[i]), float(ei[i]))
                    for i, k in enumerate(valid_keys)}

    # Per-topology top-3 by EI → ~952×3 pool
    pool = []
    for topo_id, keys in topo_cand_keys.items():
        scored = [(k, key_to_score[k]) for k in keys if k in key_to_score]
        scored.sort(key=lambda x: x[1][2], reverse=True)  # sort by EI desc
        pool.extend(scored[:BO_TOP_N_PER_TOPO])

    # Global top_n by EI
    pool.sort(key=lambda x: x[1][2], reverse=True)
    pool = pool[:top_n]

    rows = []
    for rank, (key, (m, s, e)) in enumerate(pool, start=1):
        parts = key.split("+")
        rows.append({
            "filename":           key,
            "topo":               parts[0],
            "node":               parts[1],
            "edge":               parts[2],
            "lvgp_mean":          m,
            "lvgp_std":           s,
            "ei":                 e,
            "surrogate_pred_g_L": m,   # for sim manifest compat
            "surrogate_rank":     rank, # 1=best EI
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Step 3 — Simulation submission
# ===========================================================================

def _sim_qsub(filename: str, job_idx: int, rep: int, iter_k: int,
              hpc_manifest: str, hpc_results_dir: str, hpc_log_dir: str) -> str:
    prefix = f"bois{rep}i{iter_k}"
    safe = filename[:20].replace("+", "_")
    return f"""#!/bin/bash
#PBS -N {prefix}_{safe}
#PBS -l nodes=1:ppn=1:{NODE_PROP}
#PBS -l walltime=02:00:00
#PBS -q long
#PBS -o {hpc_log_dir}/{filename}.out
#PBS -e {hpc_log_dir}/{filename}.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm2auto
export PATH="{HPC_LAMMPS}:$PATH"

{HPC_PYTHON_SIM} {HPC_GA_SRC}/mof_generation/ga_sim_worker.py \\
    --manifest {hpc_manifest} \\
    --job-index {job_idx} \\
    --output-dir {hpc_results_dir}
"""


def step_submit_sim(
    candidates: pd.DataFrame,
    rep: int,
    iter_k: int,
    local_iter_dir: str,
) -> tuple[str, str, list[str]]:
    """Write manifest, generate + upload + submit simulation qsub scripts."""
    job_prefix     = f"bois{rep}i{iter_k}"
    local_sim_dir  = os.path.join(local_iter_dir, "sim_results")
    local_qsub_dir = os.path.join(local_sim_dir, "qsub_scripts")
    local_results  = os.path.join(local_sim_dir, "results")
    hpc_sim_dir    = f"{HPC_BASE}/results/rep{rep}/iter{iter_k}/sim_results"
    hpc_qsub_dir   = f"{hpc_sim_dir}/qsub_scripts"
    hpc_log_dir    = f"{hpc_sim_dir}/logs"
    hpc_results    = f"{hpc_sim_dir}/results"
    hpc_manifest   = f"{hpc_sim_dir}/manifest.json"

    os.makedirs(local_qsub_dir, exist_ok=True)
    os.makedirs(local_results, exist_ok=True)

    # Write manifest
    jobs = []
    for _, row in candidates.iterrows():
        parts = row["filename"].split("+")
        jobs.append({
            "filename":          row["filename"],
            "topology":          parts[0],
            "node":              parts[1],
            "edge":              parts[2],
            "surrogate_pred_g_L": float(row["surrogate_pred_g_L"]),
            "surrogate_rank":    int(row["surrogate_rank"]),
        })
    manifest = {"version": 1, "n_jobs": len(jobs), "jobs": jobs}
    local_manifest = os.path.join(local_sim_dir, "manifest.json")
    with open(local_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write qsub scripts
    fnames = [j["filename"] for j in jobs]
    for idx, fn in enumerate(fnames):
        content = _sim_qsub(fn, idx, rep, iter_k, hpc_manifest, hpc_results, hpc_log_dir)
        with open(os.path.join(local_qsub_dir, f"{job_prefix}_{fn}.qsub"), "w", newline="\n") as f:
            f.write(content)

    # Create HPC dirs + upload
    _ssh_run(f"mkdir -p {hpc_qsub_dir} {hpc_log_dir} {hpc_results}")
    _rsync_up(local_sim_dir, hpc_sim_dir)

    # Submit
    remote_scripts = [f"{hpc_qsub_dir}/{job_prefix}_{fn}.qsub" for fn in fnames]
    for i in range(0, len(remote_scripts), 48):
        batch = remote_scripts[i:i + 48]
        _ssh_run(f"qas {' '.join(batch)}", check=False)
    print(f"[iter{iter_k}] Simulation: {len(jobs)} jobs submitted  prefix={job_prefix}", flush=True)

    return local_results, hpc_results, fnames


# ===========================================================================
# Step 4 — Collect simulation results → top-40 additions
# ===========================================================================

def step_collect_sim(
    local_results: str,
    candidates: pd.DataFrame,
    train_top_n: int = TRAIN_TOP_N,
) -> pd.DataFrame:
    """
    Collect sim results in EI-rank order (best EI first).
    Returns top train_top_n successful rows with real uptake + LVGP info.
    """
    cand_info = candidates.set_index("filename")
    rows = []
    for _, cand in candidates.sort_values("surrogate_rank").iterrows():
        fn = cand["filename"]
        rpath = os.path.join(local_results, fn, "result.json")
        if not os.path.isfile(rpath):
            continue
        try:
            data = json.load(open(rpath))
        except Exception:
            continue
        if data.get("status") != "success":
            continue
        uptake = data.get("real_uptake", {}).get("loading_g_L")
        if uptake is None:
            continue
        rows.append({
            "filename":       fn,
            "uptake":         float(uptake),
            "lvgp_mean":      float(cand["lvgp_mean"]),
            "lvgp_std":       float(cand["lvgp_std"]),
            "ei":             float(cand["ei"]),
            "surrogate_rank": int(cand["surrogate_rank"]),
        })
        if len(rows) >= train_top_n:
            break

    return pd.DataFrame(rows)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative LVGP-BO baseline (per rep)")
    parser.add_argument("--rep", type=int, required=True, help="Replicate index (1-5)")
    parser.add_argument("--start-iter", type=int, default=2,
                        help="First iteration to run (default: 2)")
    parser.add_argument("--end-iter", type=int, default=10,
                        help="Last iteration to run (default: 10)")
    args = parser.parse_args()

    rep = args.rep
    data_csv = os.path.join(SCRIPT_DIR, "data", f"bo_rep{rep}_iter1.csv")

    # Load random iter1 as initial training data
    accumulated = pd.read_csv(data_csv)
    print(f"[rep{rep}] Starting from random iter1: {len(accumulated)} MOFs"
          f"  mean={accumulated['uptake'].mean():.2f} g/L", flush=True)

    # Load PORMAKE topology search spaces (once)
    print(f"[rep{rep}] Loading PORMAKE topology spaces...", flush=True)
    nodes, edges, topo_list = load_pormake_dicts()
    spaces = build_topo_search_spaces(nodes, edges, topo_list)
    print(f"[rep{rep}] {len(spaces)} topologies loaded.", flush=True)

    for iter_k in range(args.start_iter, args.end_iter + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"[rep{rep}] === ITER {iter_k}  (accumulated={len(accumulated)}) ===", flush=True)
        print(f"{'='*60}", flush=True)

        local_iter_dir = os.path.join(SCRIPT_DIR, "results", f"rep{rep}", f"iter{iter_k}")
        os.makedirs(local_iter_dir, exist_ok=True)

        additions_path = os.path.join(local_iter_dir, "additions.csv")

        # Resume check
        if os.path.isfile(additions_path):
            additions = pd.read_csv(additions_path)
            if len(additions) > 0:
                print(f"[rep{rep}] iter{iter_k}: already done ({len(additions)} additions) — skipping",
                      flush=True)
                accumulated = pd.concat([accumulated, additions[["filename", "uptake"]].assign(
                    iteration=iter_k, beam="BO"
                )], ignore_index=True)
                continue

        # ── Step 1: Train LVGP ──────────────────────────────────────────────
        ckpt_path = os.path.join(local_iter_dir, "lvgp.pt")
        step_train_lvgp(accumulated, ckpt_path, rep, iter_k, local_iter_dir)

        # ── Step 2: Generate candidates via EI ─────────────────────────────
        candidates = step_generate_candidates_bo(
            ckpt_path, spaces, accumulated, rep, iter_k
        )
        if candidates.empty:
            print(f"[rep{rep}] iter{iter_k}: No BO candidates found. Skipping.", flush=True)
            continue
        print(f"[rep{rep}] iter{iter_k}: {len(candidates)} candidates for simulation", flush=True)
        candidates.to_csv(os.path.join(local_iter_dir, "candidates.csv"), index=False)

        # ── Step 3: Submit simulation ───────────────────────────────────────
        local_results, hpc_results, sim_fnames = step_submit_sim(
            candidates, rep, iter_k, local_iter_dir
        )

        # ── Step 4: Poll simulation ─────────────────────────────────────────
        sim_prefix = f"bois{rep}i{iter_k}"
        _poll_jobs(
            hpc_results_dir=hpc_results,
            local_results_dir=local_results,
            n_jobs=len(sim_fnames),
            job_prefix=sim_prefix,
            qsub_dir=f"{HPC_BASE}/results/rep{rep}/iter{iter_k}/sim_results/qsub_scripts",
            all_names=sim_fnames,
            result_subpath="result.json",
        )

        # ── Step 5: Collect sim → top-40 additions ─────────────────────────
        additions = step_collect_sim(local_results, candidates)
        n_added = len(additions)
        print(f"[rep{rep}] iter{iter_k}: {n_added} successful simulations added to training data",
              flush=True)

        if n_added == 0:
            print(f"[rep{rep}] iter{iter_k}: WARNING — 0 additions. Continuing.", flush=True)

        # Save additions
        additions.to_csv(additions_path, index=False)

        # Append to data/bo_rep{rep}.csv for cross-iter comparison
        data_out = os.path.join(SCRIPT_DIR, "data", f"bo_rep{rep}.csv")
        additions_data = additions.copy()
        additions_data["iteration"] = iter_k
        write_header = not os.path.isfile(data_out)
        additions_data.to_csv(data_out, mode="a", header=write_header, index=False)
        print(f"[rep{rep}] iter{iter_k}: appended {n_added} rows → {data_out}", flush=True)

        # Accumulate
        accumulated = pd.concat([
            accumulated,
            additions[["filename", "uptake"]].assign(iteration=iter_k, beam="BO"),
        ], ignore_index=True)

        print(f"[rep{rep}] iter{iter_k}: accumulated total = {len(accumulated)} MOFs", flush=True)

    # ── Final: save accumulated data ───────────────────────────────────────
    final_path = os.path.join(SCRIPT_DIR, "results", f"rep{rep}", "accumulated_final.csv")
    accumulated.to_csv(final_path, index=False)
    print(f"\n[rep{rep}] Done. Final accumulated: {len(accumulated)} MOFs → {final_path}",
          flush=True)


if __name__ == "__main__":
    main()
