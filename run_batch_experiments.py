# =============================================================================
# LLM2POR Batch Experiment Runner
# =============================================================================
# Non-interactive runner for multiple experiments with fixed iteration count.
# Uses feedback type 1 (4-Beam Diagnostic) for all iterations.
# =============================================================================

import os
import sys
import datetime
import json
import traceback
import argparse

# Fix Unicode encoding on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import EXPERIMENTS_DIR, ACTIVE_MODEL, validate_api_keys

from core.agent1_handler import Agent1Handler
from core.agent2_handler import Agent2Handler
from core.matchmaker import Matchmaker
from core.qmof_matchmaker import QMOFMatchmaker
from core.hmof_matchmaker import HMOFMatchmaker
from core.sensitivity_analyzer import SensitivityAnalyzer
from core.feedback_generator import FeedbackGenerator
from core.memory_manager import MemoryManager, ExperimentLogger


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

EXPERIMENTS = [
    {
        "id": 1,
        "label": "H2_Storage_PORMAKE",
        "inquiry": "Design a MOF for high capacity Hydrogen storage at 77K",
        "metric": "target",
    },
    {
        "id": 2,
        "label": "BandGap_Visible_WaterSplitting",
        "inquiry": "Design a MOF with optimal electronic band gap visible-light-driven water splitting.",
        "metric": "outputs.pbe.bandgap",
    },
    {
        "id": 3,
        "label": "BandGap_3to4eV",
        "inquiry": "Design a MOF with band gap between 3~4eV.",
        "metric": "outputs.pbe.bandgap",
    },
    {
        "id": 4,
        "label": "BandGap_UV_Activity",
        "inquiry": "Design a MOF with a band gap for UV Activity",
        "metric": "outputs.pbe.bandgap",
    },
    {
        "id": 5,
        "label": "BandGap_Below_0_1eV",
        "inquiry": "Design a MOF with band gap below 0.1eV.",
        "metric": "outputs.pbe.bandgap",
    },
    {
        "id": 6,
        "label": "BandGap_Above_4eV",
        "inquiry": "Design a MOF with a band gap above 4eV",
        "metric": "outputs.pbe.bandgap",
    },
    {
        "id": 7,
        "label": "CH4_Storage_hMOF",
        "inquiry": "Design a MOF for high methane CH4 storage capacity at 298K and 35 bar",
        "metric": "ch4_uptake_35bar_298K",
    },
    {
        "id": 8,
        "label": "CO2_Capture_hMOF",
        "inquiry": "Design a MOF for high CO2 capture capacity at low pressure (2.5 bar, 298K)",
        "metric": "co2_uptake_2_5bar_298K",
    },
    {
        "id": 9,
        "label": "XeKr_Selectivity_hMOF",
        "inquiry": "Design a MOF with high Xe/Kr selectivity for noble gas separation at 1 bar",
        "metric": "xekr_selectivity_1bar",
    },
    {
        "id": 10,
        "label": "H2_Storage_hMOF",
        "inquiry": "Design a MOF for high H2 uptake at 100 bar and 77K using hMOF database",
        "metric": "h2_uptake_100bar_77K",
    },
]

MAX_ITERATIONS = 10
FEEDBACK_TYPE = 1  # 4-Beam Diagnostic (Chemistry-First)


def run_single_experiment(exp_def: dict, batch_dir: str, strategy: str = "v231") -> dict:
    """
    Run a single experiment non-interactively.

    Returns a summary dict with status and key metrics.
    """
    label = exp_def["label"]
    inquiry = exp_def["inquiry"]
    metric = exp_def["metric"]

    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT {exp_def['id']}/10: {label}")
    print(f"# Inquiry: {inquiry}")
    print(f"# Metric: {metric}")
    print(f"# Strategy: {strategy}")
    print(f"{'#'*70}\n")

    # Set the active metric for this experiment
    config.ACTIVE_METRIC_COLUMN = metric
    # Set strategy AFTER metric is set (v230 routing depends on DB mode)
    config.set_agent1_strategy(strategy)
    is_qmof = config.is_qmof_mode()
    is_hmof = config.is_hmof_mode()
    mode_name = "QMOF" if is_qmof else "hMOF" if is_hmof else "PorMake"

    # Create experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    experiment_dir = os.path.join(batch_dir, f"exp_{timestamp}_{label}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save metadata
    with open(os.path.join(experiment_dir, "experiment_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "inquiry": inquiry,
            "metric": metric,
            "mode": mode_name,
            "model": ACTIVE_MODEL,
            "max_iterations": MAX_ITERATIONS,
            "feedback_type": FEEDBACK_TYPE,
            "strategy": strategy,
            "started": datetime.datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    # Initialize components
    agent1 = Agent1Handler()
    agent2 = Agent2Handler()

    if is_hmof:
        matchmaker = HMOFMatchmaker()
    elif is_qmof:
        matchmaker = QMOFMatchmaker()
    else:
        matchmaker = Matchmaker()

    analyzer = SensitivityAnalyzer()
    feedback_gen = FeedbackGenerator()
    memory = MemoryManager(experiment_dir, inquiry, model_name=ACTIVE_MODEL)
    logger = ExperimentLogger(experiment_dir)

    logger.log_model_info(ACTIVE_MODEL)
    logger.log_user_inquiry(inquiry)

    # Iteration loop
    current_hypothesis = None
    current_feedback = ""
    scientific_journal = []
    summary = {"label": label, "mode": mode_name, "iterations": []}

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"  [{label}] ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'='*60}")

        logger.log_iteration_start(iteration)
        iter_summary = {"iteration": iteration, "status": "ok"}

        # --- AGENT 1 ---
        try:
            if iteration == 1:
                current_hypothesis = agent1.generate_initial_hypothesis(inquiry)
            else:
                current_hypothesis = agent1.refine_hypothesis(current_feedback, scientific_journal)
        except Exception as e:
            print(f"[ERROR] Agent 1 failed: {e}")
            iter_summary["status"] = f"agent1_error: {e}"
            summary["iterations"].append(iter_summary)
            break

        if not current_hypothesis:
            print("[ERROR] Agent 1 returned empty hypothesis")
            iter_summary["status"] = "agent1_empty"
            summary["iterations"].append(iter_summary)
            break

        logger.log_hypothesis(current_hypothesis)

        # --- AGENT 2 ---
        try:
            constraints = agent2.extract_constraints(current_hypothesis)
        except Exception as e:
            print(f"[ERROR] Agent 2 failed: {e}")
            iter_summary["status"] = f"agent2_error: {e}"
            summary["iterations"].append(iter_summary)
            break

        if not constraints:
            print("[ERROR] Agent 2 returned empty constraints")
            iter_summary["status"] = "agent2_empty"
            summary["iterations"].append(iter_summary)
            break

        logger.log_constraints(constraints)

        # --- POST-EXTRACTION VALIDATION ---
        global_reqs = constraints.get('global_requirements', {})
        exclude_tags = set(global_reqs.get('exclude_tags', []))
        include_tags = set(global_reqs.get('include_tags', []))
        linker_fgs = set(constraints.get('linker_query', {}).get('functional_groups', []))

        contradiction = include_tags & exclude_tags
        if contradiction:
            for tag in contradiction:
                constraints['global_requirements']['exclude_tags'].remove(tag)

        linker_excluded = linker_fgs & exclude_tags
        if linker_excluded:
            for tag in linker_excluded:
                constraints['global_requirements']['exclude_tags'].remove(tag)

        # --- MATCHMAKER ---
        try:
            if is_hmof:
                matched_ids = matchmaker.match(constraints)
                matchmaker_results = {
                    "hmof_mode": True, "node": [], "edge": [],
                    "hmof_ids": matched_ids, "topology": [],
                    "query_specs": constraints
                }
            elif is_qmof:
                matched_ids = matchmaker.match(constraints)
                matchmaker_results = {
                    "qmof_mode": True, "node": [], "edge": [],
                    "qmof_ids": matched_ids, "topology": [],
                    "query_specs": constraints
                }
            else:
                matchmaker_results = matchmaker.smart_matchmaker_single_node(constraints)
        except Exception as e:
            print(f"[ERROR] Matchmaker failed: {e}\n{traceback.format_exc()}")
            iter_summary["status"] = f"matchmaker_error: {e}"
            summary["iterations"].append(iter_summary)
            break

        # Check results
        has_results = False
        if is_hmof:
            n_matches = len(matchmaker_results.get('hmof_ids', []))
            has_results = n_matches > 0
        elif is_qmof:
            n_matches = len(matchmaker_results.get('qmof_ids', []))
            has_results = n_matches > 0
        else:
            n_matches = len(matchmaker_results.get('edge', []))
            has_results = n_matches > 0

        iter_summary["matches"] = n_matches

        if not has_results:
            print(f"[WARNING] No candidates found. Continuing with empty results.")
            matchmaker_results = {
                "topology": [], "node": [], "edge": [],
                "qmof_ids": [], "hmof_ids": [],
                "qmof_mode": is_qmof, "hmof_mode": is_hmof,
            }

        logger.log_matchmaker_results(matchmaker_results)

        # --- SENSITIVITY ANALYSIS ---
        try:
            iter_dir = os.path.join(experiment_dir, f"iteration_{iteration}")
            sensitivity_df = analyzer.run_analysis(
                constraints, matchmaker_results,
                output_dir=iter_dir, run_id=f"iter{iteration}"
            )
        except Exception as e:
            print(f"[ERROR] Sensitivity analysis failed: {e}\n{traceback.format_exc()}")
            iter_summary["status"] = f"sensitivity_error: {e}"
            summary["iterations"].append(iter_summary)
            break

        # Save per-iteration files
        with open(os.path.join(iter_dir, "agent1_output.json"), 'w', encoding='utf-8') as f:
            json.dump(current_hypothesis, f, indent=2, ensure_ascii=False)
        with open(os.path.join(iter_dir, "agent2_output.json"), 'w', encoding='utf-8') as f:
            json.dump(constraints, f, indent=2, ensure_ascii=False)

        logger.log_sensitivity_report(sensitivity_df.to_string(index=False))

        # --- FEEDBACK ---
        try:
            current_feedback = feedback_gen.generate_feedback(
                FEEDBACK_TYPE, analyzer.filter_sets,
                metric_name=label
            )
        except Exception as e:
            print(f"[ERROR] Feedback generation failed: {e}\n{traceback.format_exc()}")
            iter_summary["status"] = f"feedback_error: {e}"
            summary["iterations"].append(iter_summary)
            break

        with open(os.path.join(iter_dir, "feedback_selected.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Feedback Type: 4-Beam Diagnostic\n{'='*50}\n\n{current_feedback}")

        logger.log_feedback_selection("4-Beam Diagnostic", current_feedback)

        # --- MEMORY ---
        sensitivity_summary = {}
        try:
            a_row = sensitivity_df[sensitivity_df['Filter'] == 'A (Chemical Only)'].iloc[0]
            d_row = sensitivity_df[sensitivity_df['Filter'] == 'D (Chem + Di + Df)'].iloc[0]
            if is_qmof:
                sensitivity_summary = {
                    "A_count": int(a_row['Count']),
                    "D_count": int(d_row['Count']),
                }
            else:
                sensitivity_summary = {
                    "A_count": int(a_row['Count']),
                    "D_count": int(d_row['Count']),
                    "A_EF_1pct": a_row.get('EF @ 1%', 'N/A'),
                    "D_EF_1pct": d_row.get('EF @ 1%', 'N/A'),
                }
            iter_summary["A_count"] = sensitivity_summary.get("A_count", 0)
            iter_summary["D_count"] = sensitivity_summary.get("D_count", 0)
        except Exception:
            pass

        memory.add_iteration(
            iteration_num=iteration, hypothesis=current_hypothesis,
            constraints=constraints, matchmaker_result=matchmaker_results,
            feedback_type="4-Beam Diagnostic", feedback_content=current_feedback,
            sensitivity_summary=sensitivity_summary
        )

        # Journal update
        lesson = current_hypothesis.get('lesson_learnt', 'No lesson recorded.')
        nodes = current_hypothesis.get('node_composition', 'Unknown Nodes')
        journal_entry = f"Iteration {iteration}: [Nodes: {nodes}] | Lesson: {lesson}"
        scientific_journal.append(journal_entry)

        logger.log_journal_update(journal_entry)
        memory.save_conversation_history(agent1.get_conversation_history())

        summary["iterations"].append(iter_summary)
        print(f"\n[{label}] Iteration {iteration} complete. Matches: {n_matches}")

    # Experiment end
    logger.log_experiment_end(len(summary["iterations"]))

    # Save summary
    summary["ended"] = datetime.datetime.now().isoformat()
    with open(os.path.join(experiment_dir, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="LLM2POR Batch Experiment Runner")
    parser.add_argument("--strategy", type=str, default="v231",
                       choices=["v229", "v230", "v231"],
                       help="Agent 1 prompt strategy (default: v231)")
    parser.add_argument("--experiments", type=str, default=None,
                       help="Comma-separated experiment IDs to run (e.g., '1,2,3'). Default: all")
    return parser.parse_args()


def main(strategy: str = "v231", experiment_ids: str = None):
    """Run experiments sequentially."""
    # Filter experiments if IDs specified
    if experiment_ids is not None:
        requested_ids = {int(x.strip()) for x in experiment_ids.split(",")}
        experiments_to_run = [e for e in EXPERIMENTS if e["id"] in requested_ids]
        if not experiments_to_run:
            print(f"[ERROR] No experiments matched IDs: {experiment_ids}")
            sys.exit(1)
    else:
        experiments_to_run = EXPERIMENTS

    print("\n" + "=" * 70)
    print("   LLM2POR BATCH EXPERIMENT RUNNER")
    print(f"   Model: {ACTIVE_MODEL}")
    print(f"   Strategy: {strategy}")
    print(f"   Experiments: {len(experiments_to_run)}")
    print(f"   Iterations per experiment: {MAX_ITERATIONS}")
    print(f"   Total LLM calls: ~{len(experiments_to_run) * MAX_ITERATIONS * 2}")
    print("=" * 70 + "\n")

    validate_api_keys()

    # Create batch directory
    batch_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    batch_dir = os.path.join(EXPERIMENTS_DIR, f"batch_{batch_ts}_{strategy}")
    os.makedirs(batch_dir, exist_ok=True)
    print(f"[Batch] Output directory: {batch_dir}\n")

    all_summaries = []

    for exp_def in experiments_to_run:
        try:
            summary = run_single_experiment(exp_def, batch_dir, strategy=strategy)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n[FATAL] Experiment {exp_def['label']} crashed: {e}")
            print(traceback.format_exc())
            all_summaries.append({
                "label": exp_def["label"],
                "status": f"fatal_error: {e}",
            })

        # Save running batch summary after each experiment
        with open(os.path.join(batch_dir, "batch_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # Final summary
    print("\n" + "=" * 70)
    print("   BATCH COMPLETE")
    print("=" * 70)
    for s in all_summaries:
        iters = s.get("iterations", [])
        status = "OK" if iters and all(i.get("status") == "ok" for i in iters) else "ISSUES"
        matches = [i.get("matches", "?") for i in iters]
        print(f"  [{status:6s}] {s['label']:35s} matches={matches}")

    print(f"\nResults saved to: {batch_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(strategy=args.strategy, experiment_ids=args.experiments)
