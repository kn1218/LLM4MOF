"""
One-shot collect step for orchestrated HPC execution.

Usage:
  python run_collect_step.py --experiment exp_dir --iteration N

Loads checkpoint + HPC results, generates feedback, saves handoff state.
Pairs with run_prepare_step.py for stepwise (prepare -> submit -> collect) HPC runs.
"""
import os, sys, json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: F401

import config
from config import EXPERIMENTS_DIR, ACTIVE_MODEL
from core.agent1_handler import Agent1Handler
from core.feedback_generator import FeedbackGenerator
from core.memory_manager import MemoryManager, ExperimentLogger
from core.live_runner import SimCache
from core.hpc.collect_results import collect_results
from core.feedback_live_adapter import live_results_to_filter_sets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment dir name")
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    experiment_dir = os.path.join(EXPERIMENTS_DIR, args.experiment)
    iteration = args.iteration
    iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")

    print(f"[Collect] Experiment: {args.experiment}, Iteration: {iteration}")

    # Load checkpoint
    ckpt_path = os.path.join(iter_dir, "checkpoint.json")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print(f"[HINT] Did you run --prepare first?")
        sys.exit(1)
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Checkpoint corrupted: {e}")
        print(f"[HINT] Delete {ckpt_path} and re-run --prepare.")
        sys.exit(1)

    hypothesis = ckpt["hypothesis"]
    constraints = ckpt["constraints"]
    conv_history = ckpt.get("conversation_history", [])
    beam_stats = ckpt.get("beam_pool_stats", {})

    # Load HPC results
    results_path = os.path.join(iter_dir, "hpc_results", "batch_results.json")
    if not os.path.exists(results_path):
        print(f"[ERROR] Results not found: {results_path}")
        sys.exit(1)

    # Parse results
    cache_path = os.path.join(experiment_dir, "sim_cache.jsonl")
    sim_cache = SimCache(cache_path)

    live_results = collect_results(
        results_path=results_path, sim_cache=sim_cache,
        n_per_beam=config.LIVE_SIM_N_PER_BEAM,
    )

    # Derive metric_name from inquiry: adsorbate keyword detection + T/P parsed from inquiry text
    import re as _re
    _ads_kw = [
        (["co2", "carbon dioxide"],                             "co2"),
        (["ch4", "methane"],                                    "ch4"),
        (["xe/kr", "xe kr", "xenon", "krypton", "selectivity"], "xekr"),
    ]
    _adsorbate = "h2"
    for _kws, _ads in _ads_kw:
        if any(kw in user_inquiry.lower() for kw in _kws):
            _adsorbate = _ads
            break
    _ads_cfg = config.LIVE_SIM_ADSORBATE_CONFIGS.get(_adsorbate, config.LIVE_SIM_ADSORBATE_CONFIGS["h2"])
    # Parse actual T/P from inquiry text (overrides config defaults, e.g. CH4 config=2.5bar but run at 35bar)
    _pres_matches = _re.findall(r'(\d+(?:\.\d+)?)\s*bar', user_inquiry, _re.IGNORECASE)
    _temp_matches = _re.findall(r'(\d+(?:\.\d+)?)\s*[Kk]\b', user_inquiry)
    _pres_bar = float(_pres_matches[0]) if _pres_matches else _ads_cfg["pressure"] / 1e5
    _temp_k = int(float(_temp_matches[0])) if _temp_matches else int(_ads_cfg["temperature"])
    _pres_str = f"{int(_pres_bar)}bar" if _pres_bar == int(_pres_bar) else f"{_pres_bar:.1f}bar"
    _ads_display = {"h2": "H2 Uptake", "ch4": "CH4 Uptake", "co2": "CO2 Uptake", "xekr": "Xe/Kr Selectivity"}
    _metric_name = f"{_ads_display[_adsorbate]} ({_pres_str} {_temp_k}K)"

    # Generate feedback
    filter_sets = live_results_to_filter_sets(live_results)
    fg = FeedbackGenerator()
    feedback = fg.generate_feedback(1, filter_sets, metric_name=_metric_name)

    print(f"\n--- Feedback Preview (first 500 chars) ---")
    print(feedback[:500])
    print("...")

    # Save feedback
    feedback_path = os.path.join(iter_dir, "feedback_selected.txt")
    with open(feedback_path, "w", encoding="utf-8") as f:
        f.write("Feedback Type: 4-Beam Diagnostic (Live Simulation)\n")
        f.write("=" * 50 + "\n\n")
        f.write(feedback)

    # Log
    logger = ExperimentLogger(experiment_dir)
    logger.log_feedback_selection("4-Beam Diagnostic (Live)", feedback)

    # Memory
    user_inquiry = ""
    inquiry_file = os.path.join(experiment_dir, "raw_user_input.txt")
    if os.path.exists(inquiry_file):
        with open(inquiry_file, "r", encoding="utf-8") as f:
            user_inquiry = f.read().strip()

    memory = MemoryManager(experiment_dir, user_inquiry, model_name=ACTIVE_MODEL)

    live_summary = {
        "n_simulations": live_results.n_real_simulations,
        "n_failures": live_results.n_failures,
        "wall_clock_s": live_results.wall_clock_seconds,
        "aborted_beams": live_results.aborted_beams,
        "cache_size": len(sim_cache),
    }

    mm_result = {
        "live_mode": True,
        "per_beam": beam_stats,
        "summary": live_summary,
    }

    memory.add_iteration(
        iteration_num=iteration,
        hypothesis=hypothesis,
        constraints=constraints,
        matchmaker_result=mm_result,
        feedback_type="4-Beam Diagnostic (Live)",
        feedback_content=feedback,
        sensitivity_summary=live_summary,
    )

    # Save handoff for next iteration
    handoff = {
        "last_iteration": iteration,
        "feedback": feedback,
        "conversation_history": conv_history,
    }
    handoff_path = os.path.join(experiment_dir, "handoff_state.json")
    with open(handoff_path, "w", encoding="utf-8") as f:
        json.dump(handoff, f, indent=2, ensure_ascii=False)

    # Save iteration summary
    iter_summary = {
        "iteration": iteration,
        "n_successes": live_results.n_real_simulations,
        "n_failures": live_results.n_failures,
        "beams": {},
    }
    for bid, beam in live_results.beams.items():
        uptakes = [s.real_uptake.get("loading_mol_kg", 0)
                   for s in beam.successes if s.real_uptake]
        iter_summary["beams"][bid] = {
            "n_success": len(beam.successes),
            "n_fail": len(beam.failures),
            "uptakes": uptakes,
            "mean_uptake": sum(uptakes) / len(uptakes) if uptakes else 0,
            "max_uptake": max(uptakes) if uptakes else 0,
        }
    summary_path = os.path.join(iter_dir, "iteration_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(iter_summary, f, indent=2)

    print(f"\n[Collect] DONE. Iteration {iteration}")
    print(f"  Successes: {live_results.n_real_simulations}")
    print(f"  Failures: {live_results.n_failures}")
    print(f"  Cache: {len(sim_cache)} entries")
    for bid, beam in live_results.beams.items():
        uptakes = [s.real_uptake.get("loading_mol_kg", 0)
                   for s in beam.successes if s.real_uptake]
        avg = sum(uptakes) / len(uptakes) if uptakes else 0
        print(f"  Beam {bid}: {len(beam.successes)} ok, avg={avg:.1f} mol/kg")

    print(f"\n__SUMMARY__:{json.dumps(iter_summary)}")


if __name__ == "__main__":
    main()
