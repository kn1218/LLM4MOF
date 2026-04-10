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
        "--prepare", action="store_true",
        help="HPC mode: run LLM agents + matchmaker, generate manifest, then exit",
    )
    parser.add_argument(
        "--collect", action="store_true",
        help="HPC mode: parse downloaded HPC results, generate feedback, continue",
    )
    return parser.parse_args()


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
        config.LIVE_SIM_SKIP_LAMMPS = True  # LAMMPS not on Windows

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

        # ----- STEP C: SIMULATION (local or HPC) -----
        if args.prepare:
            # HPC MODE: prepare manifest and exit
            print("\n[HPC Prepare] Building candidate pools...")
            beam_pools = prepare_beam_pools(
                specs=constraints,
                matchmaker=matchmaker,
                n_per_beam=config.LIVE_SIM_N_PER_BEAM,
            )

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

        memory.add_iteration(
            iteration_num=iteration,
            hypothesis=current_hypothesis,
            constraints=constraints,
            matchmaker_result={"live_mode": True, "summary": live_summary},
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
