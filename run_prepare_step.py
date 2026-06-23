"""
One-shot prepare step for orchestrated HPC execution.

Usage:
  python run_prepare_step.py --inquiry "..." [--resume exp_dir] [--raspa-cycles 10000]

Runs Agent1 -> Agent2 -> matchmaker -> manifest, saves checkpoint, exits.
Pairs with run_collect_step.py for stepwise (prepare -> submit -> collect) HPC runs.
"""
import os, sys, json, datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: F401

import config
from config import EXPERIMENTS_DIR, ACTIVE_MODEL, validate_api_keys
from core.agent1_handler import Agent1Handler
from core.agent2_handler import Agent2Handler
from core.matchmaker import Matchmaker
from core.memory_manager import MemoryManager, ExperimentLogger
from core.live_runner import prepare_beam_pools
from core.hpc.prepare_batch import prepare_manifest


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inquiry", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--raspa-cycles", type=int, default=None)
    args = parser.parse_args()

    if args.raspa_cycles:
        config.LIVE_SIM_RASPA_CYCLES = args.raspa_cycles
        config.LIVE_SIM_RASPA_INIT_CYCLES = args.raspa_cycles // 2

    validate_api_keys()

    # Experiment dir
    if args.resume:
        experiment_dir = os.path.join(EXPERIMENTS_DIR, args.resume)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        experiment_dir = os.path.join(EXPERIMENTS_DIR, f"exp_{ts}_live")
        os.makedirs(experiment_dir, exist_ok=True)

    exp_name = os.path.basename(experiment_dir)
    print(f"[Prepare] Experiment: {exp_name}")

    # Save inquiry
    with open(os.path.join(experiment_dir, "raw_user_input.txt"), "w", encoding="utf-8") as f:
        f.write(args.inquiry)

    # Determine iteration
    existing = [d for d in os.listdir(experiment_dir)
                if d.startswith("iter_") and os.path.isdir(os.path.join(experiment_dir, d))]
    if existing:
        iteration = max(int(d.split("_")[1]) for d in existing) + 1
    else:
        iteration = 1

    print(f"[Prepare] Iteration {iteration}")

    # Init components
    usage_log_path = os.path.join(experiment_dir, "usage_log.json")
    agent1 = Agent1Handler(usage_log_path=usage_log_path)
    agent2 = Agent2Handler(usage_log_path=usage_log_path)
    matchmaker = Matchmaker()
    logger = ExperimentLogger(experiment_dir)
    if iteration == 1:
        logger.log_model_info(ACTIVE_MODEL)
        logger.log_user_inquiry(args.inquiry)

    logger.log_iteration_start(iteration)

    # Restore handoff if iteration > 1
    current_feedback = ""
    if iteration > 1:
        handoff_path = os.path.join(experiment_dir, "handoff_state.json")
        if os.path.exists(handoff_path):
            with open(handoff_path, "r", encoding="utf-8") as f:
                handoff = json.load(f)
            current_feedback = handoff.get("feedback", "")
            conv = handoff.get("conversation_history", [])
            if conv:
                agent1.set_conversation_history(conv)
                print(f"[Prepare] Restored {len(conv)} conversation turns from handoff")

    # Agent 1
    if iteration == 1 and not current_feedback:
        hypothesis = agent1.generate_initial_hypothesis(args.inquiry)
    else:
        hypothesis = agent1.refine_hypothesis(current_feedback)

    if not hypothesis:
        print("[ERROR] Agent 1 failed")
        sys.exit(1)

    logger.log_hypothesis(hypothesis)

    # Agent 2
    constraints = agent2.extract_constraints(hypothesis)
    if not constraints:
        print("[ERROR] Agent 2 failed")
        sys.exit(1)

    logger.log_constraints(constraints)

    # Validation
    gr = constraints.get("global_requirements", {})
    ex = set(gr.get("exclude_tags", []))
    inc = set(gr.get("include_tags", []))
    lfg = set(constraints.get("linker_query", {}).get("functional_groups", []))
    for tag in inc & ex:
        constraints["global_requirements"]["exclude_tags"].remove(tag)
    for tag in lfg & ex:
        constraints["global_requirements"]["exclude_tags"].remove(tag)

    # Build beam pools
    print("\n[Prepare] Building candidate pools...")
    beam_pools = prepare_beam_pools(
        specs=constraints, matchmaker=matchmaker,
        n_per_beam=config.LIVE_SIM_N_PER_BEAM,
    )

    # Log matchmaker
    mm_log = {
        "topology": list({rm.component.topology for p in beam_pools.values() for rm in p}),
        "node": list({rm.component.node for p in beam_pools.values() for rm in p}),
        "edge": list({rm.component.edge for p in beam_pools.values() for rm in p}),
    }
    logger.log_matchmaker_results(mm_log)

    # Generate manifest
    iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
    manifest_path = prepare_manifest(
        beam_pools=beam_pools, iteration=iteration,
        experiment_id=exp_name, iter_dir=iter_dir,
    )

    # Save checkpoint
    checkpoint = {
        "iteration": iteration,
        "hypothesis": hypothesis,
        "constraints": constraints,
        "manifest_path": manifest_path,
        "conversation_history": agent1.get_conversation_history(),
        "beam_pool_stats": {
            bid: {"n": len(pool),
                  "topos": len(set(rm.component.topology for rm in pool)),
                  "nodes": len(set(rm.component.node for rm in pool)),
                  "edges": len(set(rm.component.edge for rm in pool))}
            for bid, pool in beam_pools.items()
        },
    }
    ckpt_path = os.path.join(iter_dir, "checkpoint.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    # Save agent outputs
    with open(os.path.join(iter_dir, "agent1_output.json"), "w", encoding="utf-8") as f:
        json.dump(hypothesis, f, indent=2, ensure_ascii=False)
    with open(os.path.join(iter_dir, "agent2_output.json"), "w", encoding="utf-8") as f:
        json.dump(constraints, f, indent=2, ensure_ascii=False)

    # Count jobs
    with open(manifest_path, "r") as f:
        n_jobs = json.load(f)["n_jobs"]

    print(f"\n[Prepare] DONE. Iteration {iteration}, {n_jobs} jobs ready.")
    print(f"[Prepare] Manifest: {manifest_path}")
    print(f"[Prepare] Experiment: {exp_name}")

    # Output machine-readable summary
    summary = {"experiment_dir": exp_name, "iteration": iteration,
               "n_jobs": n_jobs, "manifest": manifest_path, "iter_dir": iter_dir}
    print(f"\n__SUMMARY__:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
