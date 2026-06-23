# =============================================================================
# LLM4MOF Autonomous System v3 - Main Entry Point
# =============================================================================
# run_experiment.py
# Orchestrates the full autonomous MOF design loop with Agent 0 integration
# =============================================================================

import os
import sys
import re
import datetime
import json
import argparse

# Fix Unicode encoding on Windows (Korean locale cp949 can't handle Å, ², ³, etc.)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_INQUIRY, EXPERIMENTS_DIR, ACTIVE_MODEL, validate_api_keys
import config


# =============================================================================
# CLI ARGUMENT PARSING (batch/auto mode support)
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch/auto mode.

    When no arguments are provided, the experiment runs in fully interactive mode
    (original behavior). When --auto is passed, all interactive prompts are skipped.
    """
    parser = argparse.ArgumentParser(
        description="LLM4MOF Autonomous MOF Designer — Markscheme Experiment Runner",
    )
    parser.add_argument(
        "--inquiry", type=str, default=None,
        help="Design inquiry text. Skips interactive inquiry selection.",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Number of iterations to run. Skips y/n continue prompts.",
    )
    parser.add_argument(
        "--feedback-type", type=int, default=1,
        help="Deprecated. Only 4-Beam Diagnostic is supported.",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Experiment label for the directory name (e.g., 'H2_vol_100bar').",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Full auto mode: skip ALL interactive prompts. Requires --inquiry and --iterations.",
    )
    parser.add_argument(
        "--database", type=str, default=None,
        choices=["pormake", "hmof", "qmof"],
        help="Force database routing (bypasses keyword detection from query text). "
             "Use this to avoid leaking database names in the inquiry.",
    )
    parser.add_argument(
        "--agent1-prompt", type=str, default=None, dest="agent1_prompt",
        help="Override Agent 1 prompt file (bare filename resolved against prompts/, "
             "or absolute path). Used for prompt ablation studies.",
    )
    parser.add_argument(
        "--pormake-unit", type=str, default=None, dest="pormake_unit",
        choices=["volumetric", "molkg", "gperL"],
        help="Force PorMake unit variant (bypasses keyword detection). "
             "Used to avoid leaking 'gravimetric'/'mol/kg' in the inquiry.",
    )
    parser.add_argument(
        "--pormake-pressure", type=str, default=None, dest="pormake_pressure",
        choices=["5bar", "100bar"],
        help="Force PorMake pressure variant (bypasses keyword detection when --database is set).",
    )
    parser.add_argument(
        "--agent1-temp", type=float, default=None, dest="agent1_temp",
        help="Agent 1 (hypothesis) LLM temperature. Default: 0.0",
    )
    parser.add_argument(
        "--agent2-temp", type=float, default=None, dest="agent2_temp",
        help="Agent 2 (constraints) LLM temperature. Default: 0.7",
    )
    args = parser.parse_args()

    if args.auto and (args.inquiry is None or args.iterations is None):
        parser.error("--auto requires both --inquiry and --iterations")

    return args


# Module-level args (parsed once at import, accessible from run_experiment)
_CLI_ARGS = parse_args()

# Apply temperature overrides before any agent imports read config
if _CLI_ARGS.agent1_temp is not None:
    config.AGENT1_TEMPERATURE = _CLI_ARGS.agent1_temp
if _CLI_ARGS.agent2_temp is not None:
    config.AGENT2_TEMPERATURE = _CLI_ARGS.agent2_temp

from core.agent1_handler import Agent1Handler
from core.agent2_handler import Agent2Handler
from core.matchmaker import Matchmaker
from core.qmof_matchmaker import QMOFMatchmaker
from core.hmof_matchmaker import HMOFMatchmaker
from core.sensitivity_analyzer import SensitivityAnalyzer
from core.feedback_generator import FeedbackGenerator
from core.memory_manager import MemoryManager, ExperimentLogger



def print_banner():
    """Print the welcome banner."""
    print("\n" + "="*60)
    print("   LLM4MOF AUTONOMOUS MOF DESIGNER v3")
    print("   With Agent 0 + Agent1&2 ")
    print(f"   Model: {ACTIVE_MODEL}")
    print("="*60 + "\n")


def get_direct_inquiry() -> str:
    """Get the design inquiry directly from user (no Agent 0)."""
    # Auto mode: use CLI inquiry
    if _CLI_ARGS.inquiry:
        print(f"[Auto] Inquiry: {_CLI_ARGS.inquiry[:120]}...")
        return _CLI_ARGS.inquiry

    print(f"\nSelect a default inquiry or type your own:")
    print(f"  --- PorMake H2 Storage ---")
    print(f"  [1]  Volumetric, 100bar:  'Design a MOF to maximize volumetric H2 storage at 77K and 100 bar.'")
    print(f"  [2]  Volumetric, 5bar:    'Design a MOF to maximize volumetric H2 storage at 77K and 5 bar.'")
    print(f"  [3]  Gravimetric, 100bar: 'Design a MOF to maximize gravimetric H2 storage at 77K and 100 bar.'")
    print(f"  [4]  Gravimetric, 5bar:   'Design a MOF to maximize gravimetric H2 storage at 77K and 5 bar.'")
    print(f"  --- hMOF Gas Adsorption (metals: Cu/V/Zn/Zr) ---")
    print(f"  [5]  CH4 (298K, 35bar):   'Design a MOF to maximize volumetric CH4 uptake at 298K and 35 bar. Limit metal nodes to Cu, V, Zn, and Zr only.'")
    print(f"  [6]  CO2 (298K, 2.5bar):  'Design a MOF to maximize gravimetric CO2 uptake at 298K and 2.5 bar. Limit metal nodes to Cu, V, Zn, and Zr only.'")
    print(f"  [7]  Xe/Kr (273K, 1bar):  'Design a MOF to maximize Xe/Kr selectivity at 273K and 1 bar. Limit metal nodes to Cu, V, Zn, and Zr only.'")
    print(f"  [8]  H2 (77K, 100bar):    'Design a MOF to maximize gravimetric H2 uptake at 77K and 100 bar. Limit metal nodes to Cu, V, Zn, and Zr only.'")
    print(f"  --- QMOF Band Gap ---")
    print(f"  [9]  BG < 0.1 eV:         'Design a MOF with band gap below 0.1 eV.'")
    print(f"  [10] BG > 4 eV:           'Design a MOF with band gap above 4 eV.'")
    print(f"  --- Custom ---")
    print(f"  [11] Custom (type your own)")
    choice = input("Choice (1-11) > ").strip()

    # --- PorMake H2 ---
    if choice == '1' or not choice:
        return "Design a MOF to maximize volumetric H2 storage at 77K and 100 bar."
    elif choice == '2':
        return "Design a MOF to maximize volumetric H2 storage at 77K and 5 bar."
    elif choice == '3':
        return "Design a MOF to maximize gravimetric H2 storage at 77K and 100 bar."
    elif choice == '4':
        return "Design a MOF to maximize gravimetric H2 storage at 77K and 5 bar."
    # --- hMOF Gas Adsorption ---
    elif choice == '5':
        return "Design a MOF to maximize volumetric CH4 uptake at 298K and 35 bar. Limit metal nodes to Cu, V, Zn, and Zr only."
    elif choice == '6':
        return "Design a MOF to maximize gravimetric CO2 uptake at 298K and 2.5 bar. Limit metal nodes to Cu, V, Zn, and Zr only."
    elif choice == '7':
        return "Design a MOF to maximize Xe/Kr selectivity at 273K and 1 bar. Limit metal nodes to Cu, V, Zn, and Zr only."
    elif choice == '8':
        return "Design a MOF to maximize gravimetric H2 uptake at 77K and 100 bar. Limit metal nodes to Cu, V, Zn, and Zr only."
    # --- QMOF Band Gap ---
    elif choice == '9':
        return "Design a MOF with band gap below 0.1 eV."
    elif choice == '10':
        return "Design a MOF with band gap above 4 eV."
    # --- Custom ---
    else:
        custom = input("Enter your Design Inquiry > ").strip()
        return custom if custom else "Design a MOF to maximize volumetric H2 storage at 77K and 100 bar."


def feedback_type_name(choice: str = "1") -> str:
    """Return the canonical feedback type name."""
    return "4-Beam Diagnostic"


def run_experiment():
    """Main experiment loop."""
    
    print_banner()

    # 0. VALIDATE API KEYS (fail fast before any LLM call)
    try:
        validate_api_keys()
    except ValueError as e:
        print(f"\n[FATAL] {e}")
        print("[HINT] Check your .env file or ensure python-dotenv is installed: pip install python-dotenv")
        return

    # 1. GET INQUIRY (direct mode)
    user_inquiry = get_direct_inquiry()
    raw_user_query = user_inquiry
    
    print(f"\n[System] Design Inquiry: {user_inquiry[:200]}...")
    
    # --- METRIC DETECTION ---
    # Two-pass detection:
    #   Pass 1: Database/pressure routing (which dataset?)
    #   Pass 2: Unit variant for PorMake (g/L, mol/kg, or default cm³(STP)/cm³)
    _KEYWORD_MAP = [
        # hMOF gas adsorption keywords (check first — more specific)
        # NOTE: hMOF H2 requires explicit "hmof" keyword to avoid collision
        # with PorMake queries that mention "100 bar". Changed 2026-04-14.
        (["ch4", "methane"],                    "ch4_uptake_35bar_298K",    "CH4 Uptake (35bar 298K)"),
        (["co2", "carbon dioxide"],             "co2_uptake_2_5bar_298K",  "CO2 Uptake (2.5bar 298K)"),
        (["xe/kr", "xe kr", "xenon", "krypton", "selectivity"],
                                                "xekr_selectivity_1bar",   "Xe/Kr Selectivity (1bar)"),
        (["hmof", "h-mof", "hypothetical mof"], "h2_uptake_100bar_77K",    "H2 Uptake (100bar 77K)"),
        # PorMake H2 at 5 bar (same building blocks, different markscheme)
        # Use word-boundary check below instead of substring — avoids "35 bar", "2.5 bar" false positives
        # Sentinel entry: keywords list is empty; matched manually after the loop.
        ([], "target", "H2 Uptake (5bar 77K)"),
        # QMOF bandgap keywords
        (["band gap", "bandgap", "band_gap", "electronic"],
                                                "outputs.pbe.bandgap",     "Band Gap"),
        # Default H2/PORMAKE — only if no other match
        (["h2", "hydrogen"],                    "target",                  "H2 Uptake"),
    ]

    active_metric_column = config.ACTIVE_METRIC_COLUMN  # default
    active_metric_name = "H2 Uptake"
    found_metric = False
    inquiry_lower = user_inquiry.lower()

    # Override: --database flag bypasses keyword detection (avoids leaking DB names in query)
    _DB_OVERRIDE = {
        "hmof":    ("h2_uptake_100bar_77K", "H2 Uptake (100bar 77K)"),
        "qmof":    ("outputs.pbe.bandgap",  "Band Gap"),
        "pormake": ("target",               "H2 Uptake"),
    }
    if _CLI_ARGS.database and _CLI_ARGS.database in _DB_OVERRIDE:
        active_metric_column, active_metric_name = _DB_OVERRIDE[_CLI_ARGS.database]
        found_metric = True
        print(f"[System] Database override: --database {_CLI_ARGS.database} -> {active_metric_name}")
        # For hMOF, detect specific gas from query keywords
        if _CLI_ARGS.database == "hmof":
            for kws, col, name in [
                (["ch4", "methane"], "ch4_uptake_35bar_298K", "CH4 Uptake (35bar 298K)"),
                (["co2", "carbon dioxide"], "co2_uptake_2_5bar_298K", "CO2 Uptake (2.5bar 298K)"),
                (["xe/kr", "xe kr", "xenon", "krypton", "selectivity"], "xekr_selectivity_1bar", "Xe/Kr Selectivity (1bar)"),
            ]:
                if any(kw in inquiry_lower for kw in kws):
                    active_metric_column = col
                    active_metric_name = name
                    break

    # Pass 1: Database/pressure routing (skipped if --database already resolved)
    if not found_metric:
        for keywords, metric_col, display_name in _KEYWORD_MAP:
            if keywords and any(kw in inquiry_lower for kw in keywords):
                active_metric_column = metric_col
                active_metric_name = display_name
                found_metric = True
                print(f"[System] Target Metric Detected: {active_metric_name} ({metric_col})")
                break
        # H2 5bar: word-boundary match to avoid "35 bar", "2.5 bar" false positives
        if not found_metric and re.search(r'(?<!\d)5\s*bar', inquiry_lower):
            active_metric_column = "target"
            active_metric_name = "H2 Uptake (5bar 77K)"
            found_metric = True
            print(f"[System] Target Metric Detected: {active_metric_name} (target)")

    # Set config once here (single controlled mutation at startup, not at runtime)
    config.ACTIVE_METRIC_COLUMN = active_metric_column

    # Agent 1 prompt override (for ablation studies)
    if _CLI_ARGS.agent1_prompt:
        _prompt_arg = _CLI_ARGS.agent1_prompt
        if not os.path.isabs(_prompt_arg):
            _prompt_arg = os.path.join(config.PROMPTS_DIR, _prompt_arg)
        if not os.path.isfile(_prompt_arg):
            raise FileNotFoundError(f"--agent1-prompt file not found: {_prompt_arg}")
        config.AGENT1_PROMPT_PATH = _prompt_arg
        print(f"[System] Agent 1 prompt override: {os.path.basename(_prompt_arg)}")

    # Activate PorMake 5bar pressure variant if detected (keyword or CLI override)
    if _CLI_ARGS.pormake_pressure == "5bar" or "5bar 77K" in active_metric_name:
        config._PORMAKE_5BAR_ACTIVE = True
        if "(5bar 77K)" not in active_metric_name:
            active_metric_name = active_metric_name.replace("H2 Uptake", "H2 Uptake (5bar 77K)")
        print(f"[System] PorMake 5bar pressure mode active")

    # Pass 2: Unit variant detection (PorMake H2 modes only)
    # Checks for explicit unit keywords in the query.
    if active_metric_column == "target":
        # CLI override takes precedence (prevents keyword leakage in ablation studies)
        if _CLI_ARGS.pormake_unit:
            if _CLI_ARGS.pormake_unit == "gperL":
                config._PORMAKE_GPERL_ACTIVE = True
                print(f"[System] PorMake unit variant override: g/L")
            elif _CLI_ARGS.pormake_unit == "molkg":
                config._PORMAKE_GRAVIMETRIC_ACTIVE = True
                print(f"[System] PorMake unit variant override: mol/kg")
            else:  # volumetric
                print(f"[System] PorMake unit variant override: volumetric (default)")
        else:
            _UNIT_KEYWORDS = [
                (["g/l", "g per l", "gperl", "g/L", "gram per liter", "grams per liter"],  "gperL"),
                (["mol/kg", "gravimetric", "moles per kg"],                                  "molkg"),
            ]
            for unit_kws, unit_type in _UNIT_KEYWORDS:
                if any(kw in inquiry_lower for kw in unit_kws):
                    if unit_type == "gperL":
                        config._PORMAKE_GPERL_ACTIVE = True
                        print(f"[System] PorMake unit variant: g/L")
                    elif unit_type == "molkg":
                        config._PORMAKE_GRAVIMETRIC_ACTIVE = True
                        print(f"[System] PorMake unit variant: mol/kg")
                    break

    # Append unit label to metric display name so Agent 1 sees the unit in feedback
    unit_label = config.get_active_unit()
    if unit_label and f"({unit_label})" not in active_metric_name:
        active_metric_name = f"{active_metric_name} ({unit_label})"
    print(f"[System] Metric with unit: {active_metric_name}")
    print(f"[System] Markscheme CSV: {config.get_master_db_path()}")

    if not found_metric:
         print(f"[System] No specific metric detected. Defaulting to H2 Storage.")


    # 3. CREATE EXPERIMENT DIRECTORY
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if _CLI_ARGS.label:
        experiment_dir = os.path.join(EXPERIMENTS_DIR, f"exp_{timestamp}_{_CLI_ARGS.label}")
    else:
        experiment_dir = os.path.join(EXPERIMENTS_DIR, f"exp_{timestamp}_direct")
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"[System] Experiment directory: {experiment_dir}")
    
    # Save raw user input
    with open(os.path.join(experiment_dir, "raw_user_input.txt"), "w", encoding="utf-8") as f:
        f.write(raw_user_query)
    print(f"[System] Raw user input saved to raw_user_input.txt")

    # Save experiment metadata (enables batch analysis and plotting without parsing logs)
    experiment_meta = {
        "label": _CLI_ARGS.label or "direct",
        "inquiry_raw": raw_user_query,
        "inquiry_enriched": user_inquiry,
        "mode": (
            "QMOF"    if config.is_qmof_mode()    else
            "hMOF"    if config.is_hmof_mode()    else
            "PorMake"
        ),
        "model": ACTIVE_MODEL,
        "max_iterations": _CLI_ARGS.iterations or "interactive",
        "feedback_type": "4-Beam Diagnostic",
        "strategy": "v2292",
        "enriched": False,
        "metric_column": config.ACTIVE_METRIC_COLUMN,
        "metric_name": active_metric_name,
        "unit": config.get_active_unit(),
        "unit_type": config.get_active_unit_type(),
        "csv_path": config.get_master_db_path() if not (config.is_qmof_mode() or config.is_hmof_mode()) else "N/A",
        "started": datetime.datetime.now().isoformat(),
        "agent1_temperature": config.AGENT1_TEMPERATURE,
        "agent2_temperature": config.AGENT2_TEMPERATURE,
    }
    meta_path = os.path.join(experiment_dir, "experiment_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_meta, f, indent=2, ensure_ascii=False)
    print(f"[System] Experiment metadata saved to experiment_meta.json")
    
    # 4. INITIALIZE ALL COMPONENTS
    print("\n[System] Initializing components...")
    
    # --- CHECK DB EXISTS & SELECT MATCHMAKER ---
    from config import BB_DICTIONARY_V3_PATH
    is_qmof_mode = config.is_qmof_mode()
    is_hmof_mode = config.is_hmof_mode()

    if not is_qmof_mode and not is_hmof_mode and not os.path.exists(BB_DICTIONARY_V3_PATH):
        raise FileNotFoundError(
            f"V3 Building Block Database not found at {BB_DICTIONARY_V3_PATH}. "
            "Please ensure the database file exists before running."
        )

    usage_log_path = os.path.join(experiment_dir, "usage_log.json")
    agent1 = Agent1Handler(usage_log_path=usage_log_path)
    agent2 = Agent2Handler(usage_log_path=usage_log_path)

    if is_hmof_mode:
        print(f"[System] Initializing hMOF Matchmaker for gas adsorption mode ({active_metric_name})...")
        matchmaker = HMOFMatchmaker()
    elif is_qmof_mode:
        print("[System] Initializing QMOF Matchmaker for band gap mode...")
        matchmaker = QMOFMatchmaker()
    else:
        print("[System] Initializing standard Matchmaker for PORMAKE mode...")
        matchmaker = Matchmaker()

    analyzer = SensitivityAnalyzer()
    feedback_gen = FeedbackGenerator()
    feedback_gen.experiment_dir = experiment_dir
    memory = MemoryManager(experiment_dir, user_inquiry, model_name=ACTIVE_MODEL)
    logger = ExperimentLogger(experiment_dir)
    
    logger.log_model_info(ACTIVE_MODEL)
    logger.log_user_inquiry(user_inquiry)
    
    # 5. ITERATION LOOP
    iteration = 0
    current_hypothesis = None
    current_feedback = ""
    
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"                    ITERATION {iteration}")
        print(f"{'='*60}")
        
        logger.log_iteration_start(iteration)
        
        # ----- STEP A: AGENT 1 - GENERATE/REFINE HYPOTHESIS -----
        if iteration == 1:
            # First iteration: generate initial hypothesis
            current_hypothesis = agent1.generate_initial_hypothesis(user_inquiry)
        else:
            # Subsequent iterations: refine based on feedback (multi-turn context)
            current_hypothesis = agent1.refine_hypothesis(current_feedback)
        
        if not current_hypothesis:
            msg = "Agent 1 failed to generate hypothesis. Exiting."
            print(f"\n[ERROR] {msg}")
            logger.log_info(f"[ERROR] {msg}")
            break
        
        logger.log_hypothesis(current_hypothesis)
        
        # ----- STEP B: AGENT 2 - EXTRACT CONSTRAINTS -----
        constraints = agent2.extract_constraints(current_hypothesis)
        
        if not constraints:
            msg = "Agent 2 failed to extract constraints. Exiting."
            print(f"\n[ERROR] {msg}")
            logger.log_info(f"[ERROR] {msg}")
            break
        
        logger.log_constraints(constraints)
        
        # ----- STEP B.5: POST-EXTRACTION VALIDATION (Finding A fix) -----
        # Check for logical contradictions in Agent 2's output
        global_reqs = constraints.get('global_requirements', {})
        exclude_tags = set(global_reqs.get('exclude_tags', []))
        include_tags = set(global_reqs.get('include_tags', []))
        linker_fgs = set(constraints.get('linker_query', {}).get('functional_groups', []))
        
        # Contradiction: same tag in both include and exclude
        contradiction = include_tags & exclude_tags
        if contradiction:
            print(f"   [VALIDATION WARNING] Tags in BOTH include and exclude: {contradiction}")
            print(f"   [VALIDATION WARNING] Removing contradicted tags from exclude_tags.")
            for tag in contradiction:
                constraints['global_requirements']['exclude_tags'].remove(tag)
        
        # Contradiction: linker functional_group also excluded
        linker_excluded = linker_fgs & exclude_tags
        if linker_excluded:
            print(f"   [VALIDATION WARNING] Linker functional groups also in exclude_tags: {linker_excluded}")
            print(f"   [VALIDATION WARNING] Removing contradicted tags from exclude_tags.")
            for tag in linker_excluded:
                constraints['global_requirements']['exclude_tags'].remove(tag)
        
        # Log optional_tags if present (neutral — no filtering effect)
        optional_tags = global_reqs.get('optional_tags', [])
        if optional_tags:
            print(f"   [INFO] Optional tags (neutral, not filtered): {optional_tags}")
        
        # ----- STEP C: MATCHMAKER - FIND COMPONENTS -----
        print("\n[Matchmaker] Running component discovery...")
        
        # 1. Execute Matchmaker
        import traceback
        try:
            if is_hmof_mode:
                matched_ids = matchmaker.match(constraints)
                matchmaker_results = {
                    "hmof_mode": True, "node": [], "edge": [],
                    "hmof_ids": matched_ids, "topology": [],
                    "query_specs": constraints
                }
            elif is_qmof_mode:
                matched_ids = matchmaker.match(constraints)
                matchmaker_results = {"qmof_mode": True, "node": [], "edge": [], "qmof_ids": matched_ids, "topology": [], "query_specs": constraints}
            else:
                matchmaker_results = matchmaker.smart_matchmaker_single_node(constraints)
        except Exception as e:
            err_msg = f"CRITICAL ERROR in Matchmaker Execution: {e}\n{traceback.format_exc()}"
            print(f"\n[ERROR] {err_msg}")
            logger.log_matchmaker_results({"error": err_msg})
            # Break the loop to avoid cascading failures
            break

        # Check for empty results
        has_results = False
        if is_hmof_mode:
            has_results = bool(matchmaker_results.get('hmof_ids'))
        elif is_qmof_mode:
            has_results = bool(matchmaker_results.get('qmof_ids'))
        else:
            has_results = bool(matchmaker_results.get('node'))

        if matchmaker_results.get('status') == 'error' or not has_results:
            # SP-3.08: Structured error dict or empty results
            msg = matchmaker_results.get('message', 'No candidates found.')
            print(f"\n[Matchmaker] {msg}")
            logger.log_matchmaker_results({"error": msg})
            
            # Ask user if they want to continue
            print("\nNo candidates found. Agent 1 may need to adjust hypothesis.")
            if _CLI_ARGS.auto:
                print("[Auto] Continuing to next iteration (zero-result recovery).")
                choice = 'y'
            else:
                choice = input("Continue to next iteration? (y/n): ").strip().lower()
            if choice != 'y':
                break
            
            # Create empty results for feedback
            matchmaker_results = {
                "topology": [],
                "node": [],
                "edge": [],
                "qmof_ids": [],
                "hmof_ids": [],
                "qmof_mode": is_qmof_mode,
                "hmof_mode": is_hmof_mode,
            }
        
        logger.log_matchmaker_results(matchmaker_results)
        
        # ----- STEP D: SENSITIVITY ANALYSIS (Human Only) -----
        print("\n[Analysis] Running sensitivity analysis...")
        iter_dir = os.path.join(experiment_dir, f"iteration_{iteration}")
        
        sensitivity_df = analyzer.run_analysis(
            constraints, 
            matchmaker_results,
            output_dir=iter_dir,
            run_id=f"iter{iteration}"
        )
        
        # ----- SAVE PER-ITERATION FILES (as per plan) -----
        # Save agent1_output.json
        agent1_path = os.path.join(iter_dir, "agent1_output.json")
        with open(agent1_path, 'w', encoding='utf-8') as f:
            json.dump(current_hypothesis, f, indent=2, ensure_ascii=False)
        
        # Save agent2_output.json
        agent2_path = os.path.join(iter_dir, "agent2_output.json")
        with open(agent2_path, 'w', encoding='utf-8') as f:
            json.dump(constraints, f, indent=2, ensure_ascii=False)
        
        print(f"[System] Saved agent outputs to {iter_dir}")

        # Log sensitivity report (these metrics are NOT shared with Agent 1)
        logger.log_sensitivity_report(sensitivity_df.to_string(index=False))

        # Save structured beam data for plotting (beam_data.csv)
        # This enables reliable figure generation without parsing feedback text.
        import pandas as pd
        _BEAM_MAP = {'z': 'Z', 'a': 'A', 'f': 'F', 'total': 'R'}
        _BEAM_COLS = ['filename', 'target', 'di', 'df', 'sa', 'vf', 'density', 'dif', 'cv']
        beam_frames = []
        for set_key, beam_label in _BEAM_MAP.items():
            df_beam = analyzer.filter_sets.get(set_key)
            if df_beam is not None and not df_beam.empty:
                available_cols = [c for c in _BEAM_COLS if c in df_beam.columns]
                frame = df_beam[available_cols].copy()
                frame['beam_id'] = beam_label
                beam_frames.append(frame)
        if beam_frames:
            beam_data = pd.concat(beam_frames, ignore_index=True)
            beam_csv_path = os.path.join(iter_dir, "beam_data.csv")
            beam_data.to_csv(beam_csv_path, index=False, encoding='utf-8')
            print(f"[System] Saved beam_data.csv ({len(beam_data)} rows, "
                  f"beams: {beam_data['beam_id'].value_counts().to_dict()})")
        
        # ----- STEP E: FEEDBACK (4-Beam Diagnostic, always) -----
        feedback_type = 1
        if _CLI_ARGS.auto:
            if _CLI_ARGS.iterations and iteration >= _CLI_ARGS.iterations:
                print(f"\n[Auto] Reached iteration limit ({_CLI_ARGS.iterations}). Stopping.")
                break
        else:
            choice = input("\nPress Enter to continue, or type 'quit' to end: ").strip().lower()
            if choice == 'quit':
                print("\n[System] Experiment manually terminated.")
                break
        
        feedback_type_str = "4-Beam Diagnostic"
        print(f"\n[System] Feedback: {feedback_type_str}")
        
        print(f"\n[Feedback] Generating {feedback_type_str} feedback...")
        current_feedback = feedback_gen.generate_feedback(
            feedback_type, 
            analyzer.filter_sets,
            metric_name=active_metric_name
        )
        
        print(f"\n--- Feedback Preview ---")
        print(current_feedback[:500] + "..." if len(current_feedback) > 500 else current_feedback)
        
        logger.log_feedback_selection(feedback_type_str, current_feedback)
        
        # Save feedback_selected.txt (as per plan)
        feedback_path = os.path.join(iter_dir, "feedback_selected.txt")
        with open(feedback_path, 'w', encoding='utf-8') as f:
            f.write(f"Feedback Type: {feedback_type_str}\n")
            f.write("="*50 + "\n\n")
            f.write(current_feedback)
        
        # ----- STEP G: SAVE MEMORY -----
        # Create sensitivity summary for memory
        sensitivity_summary = {}
        
        try:
            a_row = sensitivity_df[sensitivity_df['Filter'] == 'A (Chemical Only)'].iloc[0]
            d_row = sensitivity_df[sensitivity_df['Filter'] == 'D (Chem + Di + Df)'].iloc[0]
            
            if is_qmof_mode:
                sensitivity_summary = {
                    "A_count": int(a_row['Count']),
                    "D_count": int(d_row['Count']),
                    "A_median_bandgap": a_row.get('Median Bandgap', 'N/A'),
                    "D_median_bandgap": d_row.get('Median Bandgap', 'N/A')
                }
            else:
                sensitivity_summary = {
                    "A_count": int(a_row['Count']),
                    "D_count": int(d_row['Count']),
                    "A_EF_1pct": a_row.get('EF @ 1%', 'N/A'),
                    "D_EF_1pct": d_row.get('EF @ 1%', 'N/A')
                }

        except Exception as e:
            print(f"[Warning] Could not extract sensitivity summary: {e}")
        
        memory.add_iteration(
            iteration_num=iteration,
            hypothesis=current_hypothesis,
            constraints=constraints,
            matchmaker_result=matchmaker_results,
            feedback_type=feedback_type_str,
            feedback_content=current_feedback,
            sensitivity_summary=sensitivity_summary
        )
        
        # Save conversation history
        memory.save_conversation_history(agent1.get_conversation_history())
        
        print(f"\n[System] Iteration {iteration} complete.")
    
    # 6. EXPERIMENT END
    logger.log_experiment_end(iteration)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Mode: Direct (No Agent 0)")
    print(f"Total Iterations: {iteration}")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_experiment()
