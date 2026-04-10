# =============================================================================
# LLM2POR Autonomous System v3 - Live Simulation Entry Point
# =============================================================================
# run_live_experiment.py
# Mirrors run_experiment.py but replaces the markscheme/sensitivity-analyzer
# path with real PORMAKE → LAMMPS → RASPA3 simulations.
#
# Usage:
#   conda activate llm2auto
#   python run_live_experiment.py
#   python run_live_experiment.py --resume <runid>
#   python run_live_experiment.py --smoke   (1/beam, 200 cycles, quick validation)
# =============================================================================

import os
import sys
import datetime
import json
import argparse
import subprocess
import time

# Fix Unicode encoding on Windows (Korean locale cp949 can't handle Å, ², ³)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # Prevent OpenMP duplicate lib crash (torch + numpy on Windows)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torch must be imported before any module that transitively loads it
# (filter_candidate → torch) to avoid Windows DLL search-order issues.
import torch  # noqa: F401

from config import DEFAULT_INQUIRY, EXPERIMENTS_DIR, ACTIVE_MODEL, validate_api_keys
import config

from core.agent0_handler import Agent0Handler
from core.agent1_handler import Agent1Handler
from core.agent2_handler import Agent2Handler
from core.matchmaker import Matchmaker
from core.feedback_generator import FeedbackGenerator
from core.memory_manager import MemoryManager, ExperimentLogger
from core.live_runner import run_live_iteration, prepare_beam_pools, SimCache
from core.feedback_live_adapter import live_results_to_filter_sets
from core.hpc.prepare_batch import prepare_manifest
from core.hpc.collect_results import collect_results


def print_banner(smoke: bool = False) -> None:
    mode_label = "SMOKE TEST" if smoke else "LIVE SIMULATION"
    print("\n" + "=" * 60)
    print(f"   LLM2POR AUTONOMOUS MOF DESIGNER v3 — {mode_label}")
    print(f"   Model: {ACTIVE_MODEL}")
    print(f"   Beams: {config.LIVE_SIM_N_BEAMS} × "
          f"{config.LIVE_SIM_N_PER_BEAM} successes/beam")
    print(f"   RASPA cycles: {config.LIVE_SIM_RASPA_CYCLES} prod + "
          f"{config.LIVE_SIM_RASPA_INIT_CYCLES} init")
    print(f"   LAMMPS: {'ON' if not config.LIVE_SIM_SKIP_LAMMPS else 'SKIP'}")
    print(f"   Max iterations: {config.LIVE_SIM_MAX_ITERATIONS}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM2POR Live Simulation Experiment"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume a previous run by its experiment directory name",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 1/beam, 200 RASPA cycles, 1 iteration",
    )
    parser.add_argument(
        "--inquiry", type=str, default=None,
        help="Direct inquiry string (skip interactive prompt)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override max iterations",
    )
    parser.add_argument(
        "--hpc", action="store_true",
        help="HPC mode: single-process loop with SSH polling (production)",
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="HPC manual mode: run LLM agents + matchmaker, generate manifest, then exit",
    )
    parser.add_argument(
        "--collect", action="store_true",
        help="HPC manual mode: parse downloaded HPC results, generate feedback, continue",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HPC SSH helpers
# ---------------------------------------------------------------------------

def _ssh_run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a command on HPC via SSH with retry logic.

    Uses config.HPC_HOST and retries on connection failure with
    exponential backoff (config.HPC_SSH_RETRY_DELAYS).
    """
    ssh_cmd = ["ssh", config.HPC_HOST, cmd]
    delays = config.HPC_SSH_RETRY_DELAYS

    for attempt in range(config.HPC_SSH_RETRIES):
        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 or not check:
                return result
            # Non-zero exit but not a connection error — return as-is
            if result.returncode != 255:
                return result
            # SSH error (rc=255) — retry
            print(f"   [SSH] Connection error (attempt {attempt + 1}/"
                  f"{config.HPC_SSH_RETRIES}): {result.stderr.strip()[:100]}")
        except subprocess.TimeoutExpired:
            print(f"   [SSH] Timeout (attempt {attempt + 1}/{config.HPC_SSH_RETRIES})")

        if attempt < config.HPC_SSH_RETRIES - 1:
            delay = delays[min(attempt, len(delays) - 1)]
            print(f"   [SSH] Retrying in {delay}s...")
            time.sleep(delay)

    raise ConnectionError(
        f"SSH to {config.HPC_HOST} failed after {config.HPC_SSH_RETRIES} attempts"
    )


def _scp_upload(local_path: str, remote_path: str) -> None:
    """Upload a file to HPC via scp."""
    subprocess.run(
        ["scp", local_path, f"{config.HPC_HOST}:{remote_path}"],
        check=True, capture_output=True, text=True, timeout=120,
    )


def _scp_download(remote_path: str, local_path: str) -> None:
    """Download a file from HPC via scp."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    subprocess.run(
        ["scp", f"{config.HPC_HOST}:{remote_path}", local_path],
        check=True, capture_output=True, text=True, timeout=120,
    )


def _hpc_simulate(
    beam_pools: dict,
    iteration: int,
    experiment_id: str,
    experiment_dir: str,
    sim_cache,
) -> "LiveResults":
    """
    Run one iteration's simulations on HPC via SSH polling.

    Single-process flow:
      1. Generate manifest from beam_pools
      2. scp upload manifest + HPC scripts to dirac1
      3. ssh submit via submit_iteration.sh
      4. Poll via SSH every HPC_POLL_INTERVAL until all .DONE files appear
      5. ssh run aggregate_results.py on HPC
      6. scp download batch_results.json
      7. Parse into LiveResults via collect_results()
    """
    iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)

    # Step 1: Generate manifest
    manifest_path = prepare_manifest(
        beam_pools=beam_pools,
        iteration=iteration,
        experiment_id=experiment_id,
        iter_dir=iter_dir,
    )

    # Count total jobs
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    n_jobs = manifest_data["n_jobs"]

    if n_jobs == 0:
        print("[HPC] No jobs to simulate (all beams empty)")
        from core.live_runner import LiveResults
        return LiveResults()

    # Step 2: Upload to HPC
    hpc_base = config.HPC_BASE_DIR
    hpc_exp_dir = f"{hpc_base}/{experiment_id}"
    hpc_iter_dir = f"{hpc_exp_dir}/iter_{iteration}"

    print(f"\n[HPC] Uploading manifest ({n_jobs} jobs) to {config.HPC_HOST}...")
    _ssh_run(f"mkdir -p {hpc_iter_dir}/results")
    _scp_upload(manifest_path, f"{hpc_iter_dir}/batch_manifest.json")

    # Upload HPC scripts (ensure they're current)
    hpc_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpc")
    hpc_remote_scripts = f"{hpc_base}/hpc"
    _ssh_run(f"mkdir -p {hpc_remote_scripts}")
    for script in ["run_mof_sim.py", "submit_iteration.sh", "aggregate_results.py"]:
        script_path = os.path.join(hpc_scripts_dir, script)
        if os.path.exists(script_path):
            _scp_upload(script_path, f"{hpc_remote_scripts}/{script}")

    # Step 3: Submit jobs
    print(f"[HPC] Submitting {n_jobs} jobs via qas...")
    submit_result = _ssh_run(
        f"cd {hpc_iter_dir} && bash {hpc_remote_scripts}/submit_iteration.sh "
        f"batch_manifest.json results {config.HPC_NODE_PROPERTY}"
    )
    print(f"[HPC] Submit output: {submit_result.stdout.strip()[-200:]}")
    if submit_result.returncode != 0:
        print(f"[HPC] Submit stderr: {submit_result.stderr.strip()[-200:]}")
        raise RuntimeError(f"Job submission failed (rc={submit_result.returncode})")

    # Step 4: Poll for completion
    print(f"[HPC] Polling every {config.HPC_POLL_INTERVAL}s for {n_jobs} .DONE files...")
    max_polls = int(config.HPC_POLL_MAX_HOURS * 3600 / config.HPC_POLL_INTERVAL)
    results_dir = f"{hpc_iter_dir}/results"

    for poll in range(1, max_polls + 1):
        time.sleep(config.HPC_POLL_INTERVAL)

        count_result = _ssh_run(
            f"ls {results_dir}/*.DONE 2>/dev/null | wc -l",
            check=False,
        )
        try:
            n_done = int(count_result.stdout.strip())
        except ValueError:
            n_done = 0

        elapsed_min = poll * config.HPC_POLL_INTERVAL / 60
        print(f"   [HPC Poll #{poll}] {n_done}/{n_jobs} complete "
              f"({elapsed_min:.0f} min elapsed)")

        if n_done >= n_jobs:
            print(f"[HPC] All {n_jobs} jobs complete!")
            break
    else:
        print(f"[HPC WARNING] Timeout after {config.HPC_POLL_MAX_HOURS}h "
              f"({n_done}/{n_jobs} complete). Proceeding with partial results.")

    # Step 5: Aggregate results on HPC
    print("[HPC] Aggregating results on remote...")
    agg_result = _ssh_run(
        f"cd {hpc_iter_dir} && python {hpc_remote_scripts}/aggregate_results.py "
        f"--manifest batch_manifest.json --results-dir results"
    )
    print(f"[HPC] Aggregate: {agg_result.stdout.strip()[-200:]}")

    # Step 6: Download batch_results.json
    local_results_dir = os.path.join(iter_dir, "hpc_results")
    os.makedirs(local_results_dir, exist_ok=True)
    local_results_path = os.path.join(local_results_dir, "batch_results.json")

    print("[HPC] Downloading results...")
    _scp_download(
        f"{results_dir}/batch_results.json",
        local_results_path,
    )

    # Step 7: Parse into LiveResults
    print("[HPC] Parsing results...")
    live_results = collect_results(
        results_path=local_results_path,
        sim_cache=sim_cache,
        n_per_beam=config.LIVE_SIM_N_PER_BEAM,
    )

    return live_results


def run_live_experiment() -> None:
    args = parse_args()

    # --- Smoke test overrides ---
    if args.smoke:
        config.LIVE_SIM_N_PER_BEAM = 1
        config.LIVE_SIM_RASPA_CYCLES = 200
        config.LIVE_SIM_RASPA_INIT_CYCLES = 100
        config.LIVE_SIM_MAX_ITERATIONS = 1
        config.LIVE_SIM_POOL_MULTIPLIER = 2
        config.LIVE_SIM_POOL_MULTIPLIER_RANDOM = 2
        if not args.hpc:
            config.LIVE_SIM_SKIP_LAMMPS = True  # LAMMPS not on Windows (HPC has it)

    if args.iterations:
        config.LIVE_SIM_MAX_ITERATIONS = args.iterations

    print_banner(smoke=args.smoke)

    # --- Validate API keys ---
    validate_api_keys()

    # --- Get user inquiry ---
    if args.inquiry:
        user_inquiry = args.inquiry
    elif args.resume:
        # Load from previous experiment
        resume_dir = os.path.join(EXPERIMENTS_DIR, args.resume)
        inquiry_file = os.path.join(resume_dir, "raw_user_input.txt")
        if os.path.exists(inquiry_file):
            with open(inquiry_file, "r", encoding="utf-8") as f:
                user_inquiry = f.read().strip()
        else:
            user_inquiry = input("Enter design inquiry: ").strip()
    else:
        print("Enter your MOF design inquiry (or press Enter for default):")
        user_inquiry = input("> ").strip()
        if not user_inquiry:
            user_inquiry = DEFAULT_INQUIRY

    print(f"\n[System] Design Inquiry: {user_inquiry[:200]}...")

    # --- Metric detection (same as run_experiment.py) ---
    active_metric_name = "H2 Uptake"
    config.ACTIVE_METRIC_COLUMN = "target"
    print(f"[System] Live mode: metric = {active_metric_name} (RASPA3 GCMC)")

    # --- Create/resume experiment directory ---
    if args.resume:
        experiment_dir = os.path.join(EXPERIMENTS_DIR, args.resume)
        if not os.path.isdir(experiment_dir):
            print(f"[ERROR] Resume directory not found: {experiment_dir}")
            sys.exit(1)
        print(f"[System] Resuming experiment: {experiment_dir}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        mode_label = "smoke" if args.smoke else "live"
        experiment_dir = os.path.join(
            EXPERIMENTS_DIR, f"exp_{timestamp}_{mode_label}"
        )
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"[System] Experiment directory: {experiment_dir}")

    # Save raw input
    with open(
        os.path.join(experiment_dir, "raw_user_input.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(user_inquiry)

    # --- Initialize components ---
    print("\n[System] Initializing components...")
    agent1 = Agent1Handler()
    agent2 = Agent2Handler()
    matchmaker = Matchmaker()
    feedback_gen = FeedbackGenerator()
    memory = MemoryManager(
        experiment_dir, user_inquiry, model_name=ACTIVE_MODEL
    )
    logger = ExperimentLogger(experiment_dir)
    logger.log_model_info(ACTIVE_MODEL)
    logger.log_user_inquiry(user_inquiry)

    # --- Simulation cache ---
    cache_path = os.path.join(experiment_dir, "sim_cache.jsonl")
    sim_cache = SimCache(cache_path)

    # --- Determine starting iteration (for resume) ---
    start_iter = 1
    if args.resume:
        existing = [
            d for d in os.listdir(experiment_dir)
            if d.startswith("iter_") and os.path.isdir(
                os.path.join(experiment_dir, d)
            )
        ]
        if existing:
            last_iter = max(int(d.split("_")[1]) for d in existing)
            if args.collect:
                # --collect: resume at the iteration with a checkpoint
                start_iter = last_iter
            else:
                start_iter = last_iter + 1
            print(f"[System] Resuming from iteration {start_iter} "
                  f"(cache: {len(sim_cache)} entries)")

    # --- ITERATION LOOP ---
    current_hypothesis = None
    current_feedback = ""
    max_iter = config.LIVE_SIM_MAX_ITERATIONS

    for iteration in range(start_iter, start_iter + max_iter):
        print(f"\n{'='*60}")
        print(f"              LIVE ITERATION {iteration}")
        print(f"{'='*60}")

        logger.log_iteration_start(iteration)

        # ----- STEP A/B: LLM AGENTS (skip if --collect with checkpoint) -----
        if args.collect:
            # Load checkpoint from --prepare phase
            iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
            ckpt_path = os.path.join(iter_dir, "checkpoint.json")
            if os.path.exists(ckpt_path):
                print(f"[Collect] Loading checkpoint: {ckpt_path}")
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                current_hypothesis = ckpt["hypothesis"]
                constraints = ckpt["constraints"]
            else:
                print(f"[ERROR] No checkpoint at {ckpt_path}. Run --prepare first.")
                break
        else:
            # ----- STEP A: AGENT 1 — GENERATE/REFINE HYPOTHESIS -----
            # Restore handoff state from previous --collect (feedback + conversation)
            if iteration > 1 and not current_feedback:
                handoff_path = os.path.join(experiment_dir, "handoff_state.json")
                if os.path.exists(handoff_path):
                    print(f"[Handoff] Restoring Agent 1 state from {handoff_path}")
                    with open(handoff_path, "r", encoding="utf-8") as f:
                        handoff = json.load(f)
                    current_feedback = handoff.get("feedback", "")
                    conv_history = handoff.get("conversation_history", [])
                    if conv_history:
                        agent1.set_conversation_history(conv_history)
                        print(f"[Handoff] Restored {len(conv_history)} conversation turns")

            if iteration == 1 and not current_feedback:
                current_hypothesis = agent1.generate_initial_hypothesis(user_inquiry)
            else:
                current_hypothesis = agent1.refine_hypothesis(current_feedback)

            if not current_hypothesis:
                print("[ERROR] Agent 1 failed to generate hypothesis. Exiting.")
                break

            logger.log_hypothesis(current_hypothesis)

            # ----- STEP B: AGENT 2 — EXTRACT CONSTRAINTS -----
            constraints = agent2.extract_constraints(current_hypothesis)
            if not constraints:
                print("[ERROR] Agent 2 failed to extract constraints. Exiting.")
                break

            logger.log_constraints(constraints)

        # ----- STEP B.5: VALIDATION (same as run_experiment.py) -----
        global_reqs = constraints.get("global_requirements", {})
        exclude_tags = set(global_reqs.get("exclude_tags", []))
        include_tags = set(global_reqs.get("include_tags", []))
        linker_fgs = set(
            constraints.get("linker_query", {}).get("functional_groups", [])
        )

        for tag in include_tags & exclude_tags:
            constraints["global_requirements"]["exclude_tags"].remove(tag)
        for tag in linker_fgs & exclude_tags:
            constraints["global_requirements"]["exclude_tags"].remove(tag)

        # ----- STEP C: SIMULATION (local, HPC auto, or HPC manual) -----

        # Build beam pools (needed for --hpc and --prepare)
        if args.hpc or args.prepare:
            print("\n[HPC] Building candidate pools...")
            beam_pools = prepare_beam_pools(
                specs=constraints,
                matchmaker=matchmaker,
                n_per_beam=config.LIVE_SIM_N_PER_BEAM,
            )

            # Collect matchmaker stats for logging (Fix 1.2 + 1.3)
            matchmaker_summary = {}
            for bid, pool in beam_pools.items():
                topos = set()
                nodes = set()
                edges = set()
                for rm in pool:
                    topos.add(rm.component.topology)
                    nodes.add(rm.component.node)
                    edges.add(rm.component.edge)
                matchmaker_summary[bid] = {
                    "n_candidates": len(pool),
                    "n_topologies": len(topos),
                    "n_nodes": len(nodes),
                    "n_edges": len(edges),
                }
            total_candidates = sum(v["n_candidates"] for v in matchmaker_summary.values())
            print(f"[HPC] Total candidates across beams: {total_candidates}")

            # Log matchmaker results (Fix 1.3)
            mm_log = {
                "topology": list({rm.component.topology for p in beam_pools.values() for rm in p}),
                "node": list({rm.component.node for p in beam_pools.values() for rm in p}),
                "edge": list({rm.component.edge for p in beam_pools.values() for rm in p}),
            }
            logger.log_matchmaker_results(mm_log)

        if args.hpc:
            # HPC AUTO MODE: single-process SSH polling
            experiment_id = os.path.basename(experiment_dir)
            live_results = _hpc_simulate(
                beam_pools=beam_pools,
                iteration=iteration,
                experiment_id=experiment_id,
                experiment_dir=experiment_dir,
                sim_cache=sim_cache,
            )

        elif args.prepare:
            # HPC MANUAL MODE: prepare manifest and exit
            iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
            experiment_id = os.path.basename(experiment_dir)
            manifest_path = prepare_manifest(
                beam_pools=beam_pools,
                iteration=iteration,
                experiment_id=experiment_id,
                iter_dir=iter_dir,
            )

            # Save checkpoint for --collect phase
            checkpoint = {
                "iteration": iteration,
                "hypothesis": current_hypothesis,
                "constraints": constraints,
                "manifest_path": manifest_path,
                "experiment_dir": experiment_dir,
            }
            ckpt_path = os.path.join(iter_dir, "checkpoint.json")
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            print(f"\n[HPC Prepare] Manifest: {manifest_path}")
            print(f"[HPC Prepare] Checkpoint: {ckpt_path}")
            print(f"\n[ACTION] Upload manifest to HPC and submit jobs.")
            print(f"  Or use: bash scripts/autonomous_loop.sh {experiment_id}")
            break  # Exit — resume with --collect after HPC completes

        elif args.collect:
            # HPC MODE: parse downloaded results
            iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
            results_path = os.path.join(iter_dir, "hpc_results", "batch_results.json")

            if not os.path.exists(results_path):
                print(f"[ERROR] Results not found: {results_path}")
                print("Download batch_results.json from HPC first.")
                break

            print(f"\n[HPC Collect] Parsing results from {results_path}")
            live_results = collect_results(
                results_path=results_path,
                sim_cache=sim_cache,
                n_per_beam=config.LIVE_SIM_N_PER_BEAM,
            )

        else:
            # LOCAL MODE: run simulation directly
            print("\n[Live Runner] Starting live simulation iteration...")

            live_results = run_live_iteration(
                specs=constraints,
                matchmaker=matchmaker,
                iteration=iteration,
                run_dir=experiment_dir,
                sim_cache=sim_cache,
                n_per_beam=config.LIVE_SIM_N_PER_BEAM,
            )

        # Check for catastrophic failure — only count beams that had
        # simulation failures (pool exhausted), NOT matchmaker empty-results
        # (which are normal for over-constrained first hypotheses).
        sim_aborted = [
            b for b in live_results.aborted_beams
            if b in live_results.beams and live_results.beams[b].pool_size > 0
        ]
        if len(sim_aborted) >= 3:
            print(f"\n[WARNING] {len(sim_aborted)} beams failed during simulation: "
                  f"{sim_aborted}")
            print("Too many simulation failures. Stopping experiment.")
            break
        if live_results.aborted_beams:
            mm_empty = [b for b in live_results.aborted_beams if b not in sim_aborted]
            if mm_empty:
                print(f"\n[INFO] Beams {mm_empty} had no matchmaker candidates "
                      f"(over-constrained hypothesis — normal for iteration 1)")

        # ----- SAVE PER-ITERATION FILES -----
        iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        agent1_path = os.path.join(iter_dir, "agent1_output.json")
        with open(agent1_path, "w", encoding="utf-8") as f:
            json.dump(current_hypothesis, f, indent=2, ensure_ascii=False)

        agent2_path = os.path.join(iter_dir, "agent2_output.json")
        with open(agent2_path, "w", encoding="utf-8") as f:
            json.dump(constraints, f, indent=2, ensure_ascii=False)

        # ----- STEP D: FEEDBACK GENERATION -----
        # Convert live results → filter_sets format → feedback type 1 (4-beam)
        filter_sets = live_results_to_filter_sets(live_results)

        feedback_type = 1  # Always 4-beam diagnostic in live mode
        current_feedback = feedback_gen.generate_feedback(
            feedback_type, filter_sets, metric_name=active_metric_name
        )

        print(f"\n--- Feedback Preview ---")
        preview = current_feedback[:500]
        print(preview + "..." if len(current_feedback) > 500 else preview)

        # Save feedback
        feedback_path = os.path.join(iter_dir, "feedback_selected.txt")
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write("Feedback Type: 4-Beam Diagnostic (Live Simulation)\n")
            f.write("=" * 50 + "\n\n")
            f.write(current_feedback)

        logger.log_feedback_selection("4-Beam Diagnostic (Live)", current_feedback)

        # ----- STEP E: MEMORY -----
        live_summary = {
            "n_simulations": live_results.n_real_simulations,
            "n_failures": live_results.n_failures,
            "wall_clock_s": live_results.wall_clock_seconds,
            "aborted_beams": live_results.aborted_beams,
            "cache_size": len(sim_cache),
        }

        # Build matchmaker_result with real counts (Fix 1.2)
        if args.hpc or args.prepare:
            # matchmaker_summary was computed during beam pool construction
            mm_result = {
                "live_mode": True,
                "topology": list({rm.component.topology for p in beam_pools.values() for rm in p}),
                "node": list({rm.component.node for p in beam_pools.values() for rm in p}),
                "edge": list({rm.component.edge for p in beam_pools.values() for rm in p}),
                "per_beam": matchmaker_summary,
                "summary": live_summary,
            }
        else:
            mm_result = {"live_mode": True, "summary": live_summary}

        memory.add_iteration(
            iteration_num=iteration,
            hypothesis=current_hypothesis,
            constraints=constraints,
            matchmaker_result=mm_result,
            feedback_type="4-Beam Diagnostic (Live)",
            feedback_content=current_feedback,
            sensitivity_summary=live_summary,
        )

        memory.save_conversation_history(agent1.get_conversation_history())

        # ----- SAVE CROSS-ITERATION STATE (for --prepare/--collect handoff) -----
        # This enables the next --prepare to restore Agent 1's context
        handoff_path = os.path.join(experiment_dir, "handoff_state.json")
        with open(handoff_path, "w", encoding="utf-8") as f:
            json.dump({
                "last_iteration": iteration,
                "feedback": current_feedback,
                "conversation_history": agent1.get_conversation_history(),
            }, f, indent=2, ensure_ascii=False)
        print(f"[System] Saved handoff state for next iteration")

        print(f"\n[System] Iteration {iteration} complete. "
              f"({live_results.wall_clock_seconds:.0f}s, "
              f"cache={len(sim_cache)})")

    # --- EXPERIMENT END ---
    logger.log_experiment_end(iteration)

    print(f"\n{'='*60}")
    print("LIVE EXPERIMENT COMPLETE")
    print(f"Total Iterations: {iteration}")
    print(f"Simulation Cache: {len(sim_cache)} entries")
    print(f"Experiment Dir: {experiment_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_live_experiment()
