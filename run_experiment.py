# =============================================================================
# LLM2POR Autonomous System v3 - Main Entry Point
# =============================================================================
# run_experiment.py
# Orchestrates the full autonomous MOF design loop with Agent 0 integration
# =============================================================================

import os
import sys
import datetime
import json

# Fix Unicode encoding on Windows (Korean locale cp949 can't handle Å, ², ³, etc.)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_INQUIRY, EXPERIMENTS_DIR, ACTIVE_MODEL, validate_api_keys
import config

from core.agent0_handler import Agent0Handler
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
    print("   LLM2POR AUTONOMOUS MOF DESIGNER v3")
    print("   With Agent 0 + Agent1&2 ")
    print(f"   Model: {ACTIVE_MODEL}")
    print("="*60 + "\n")


def get_mode_choice() -> str:
    """Ask user if they want to use Agent 0 or direct inquiry."""
    print("Select Experiment Mode:")
    print("[1] With Agent 0 - Interactive problem formulation (recommended)")
    print("[2] Direct Inquiry - Skip to Agent 1 (baseline comparison)")
    print("-"*60)
    
    while True:
        choice = input("Your choice: ").strip()
        if choice in ['1', '2']:
            return choice
        print("Invalid choice. Please enter 1 or 2.")


def run_agent0_conversation() -> tuple:
    """
    Run Agent 0 conversation and return problem spec and enriched inquiry.
    
    Returns:
        (problem_spec_json, enriched_inquiry_text, initial_query) or (None, None, None) if cancelled
    """
    print("\n" + "="*60)
    print("AGENT 0: PROBLEM FORMULATION")
    print("="*60)
    print("Agent 0 will help clarify your requirements.")
    print("Type 'proceed' anytime to skip remaining questions.")
    print("Type 'quit' to cancel.")
    print("="*60 + "\n")
    
    # Get initial query
    initial_query = input("What MOF do you want to design? → ").strip()
    
    if initial_query.lower() == 'quit':
        return None, None, None
    
    if not initial_query:
        initial_query = DEFAULT_INQUIRY
    
    # Initialize Agent 0
    agent0 = Agent0Handler()
    
    # Start conversation
    response = agent0.start_conversation(initial_query)
    print(f"\n[Agent 0]: {response}\n")
    
    # Continue until complete or user quits
    max_turns = config.AGENT0_MAX_TURNS
    turn = 0
    
    while not agent0.is_complete() and turn < max_turns:
        user_input = input("Your response → ").strip()
        
        if user_input.lower() == 'quit':
            print("\n[System] Interview cancelled.")
            return None, None, None
        
        response = agent0.continue_conversation(user_input)
        print(f"\n[Agent 0]: {response}\n")
        turn += 1
    
    if agent0.is_complete():
        problem_spec = agent0.get_problem_spec()
        enriched_inquiry = agent0.get_user_inquiry_text()
        
        print("\n" + "="*60)
        print("PROBLEM SPECIFICATION COMPLETE")
        print("="*60)
        print(f"\nEnriched inquiry for Agent 1:\n{enriched_inquiry}")
        
        return problem_spec, enriched_inquiry, initial_query
    else:
        print("\n[System] Max turns reached. Using partial specification.")
        return agent0.get_problem_spec(), agent0.get_user_inquiry_text(), initial_query


DEFAULT_INQUIRY_BANDGAP = "Design a MOF with optimal electronic band gap visible-light-driven water splitting."

def get_direct_inquiry() -> str:
    """Get the design inquiry directly from user (no Agent 0)."""
    print(f"\nSelect a default inquiry or type your own:")
    print(f"  [1] H2 Storage (PORMAKE): '{DEFAULT_INQUIRY}'")
    print(f"  [2] Band Gap (Visible/Water Splitting): '{DEFAULT_INQUIRY_BANDGAP}'")
    print(f"  [3] Band Gap (3~4eV): 'Design a MOF with band gap between 3~4eV.'")
    print(f"  [4] Band Gap (UV Activity): 'Design a MOF with a band gap for UV Activity'")
    print(f"  [5] Band Gap (<0.1eV): 'Design a MOF with band gap below 0.1eV.'")
    print(f"  [6] Band Gap (>4eV): 'Design a MOF with a band gap above 4eV'")
    print(f"  --- hMOF Gas Adsorption (51K hypothetical MOFs) ---")
    print(f"  [7] CH4 Storage: 'Design a MOF for high methane storage at 298K'")
    print(f"  [8] CO2 Capture: 'Design a MOF for CO2 capture at low pressure'")
    print(f"  [9] Xe/Kr Selectivity: 'Design a MOF for high Xe/Kr selectivity'")
    print(f"  [10] H2 Storage (hMOF): 'Design a MOF for high H2 uptake at 100 bar 77K'")
    print(f"  [11] Custom (type your own)")
    choice = input("Choice (1-11) > ").strip()
    
    if choice == '1' or not choice:
        return DEFAULT_INQUIRY
    elif choice == '2':
        return DEFAULT_INQUIRY_BANDGAP
    elif choice == '3':
        return "Design a MOF with band gap between 3~4eV."
    elif choice == '4':
        return "Design a MOF with a band gap for UV Activity"
    elif choice == '5':
        return "Design a MOF with band gap below 0.1eV."
    elif choice == '6':
        return "Design a MOF with a band gap above 4eV"
    elif choice == '7':
        return "Design a MOF for high methane CH4 storage capacity at 298K and 35 bar"
    elif choice == '8':
        return "Design a MOF for high CO2 capture capacity at low pressure (2.5 bar, 298K)"
    elif choice == '9':
        return "Design a MOF with high Xe/Kr selectivity for noble gas separation at 1 bar"
    elif choice == '10':
        return "Design a MOF for high H2 uptake at 100 bar and 77K using hMOF database"
    else:
        custom = input("Enter your Design Inquiry > ").strip()
        return custom if custom else DEFAULT_INQUIRY


def feedback_type_name(choice: str) -> str:
    """Convert choice to feedback type name."""
    names = {
        '1': '3-Beam Diagnostic',
        '2': 'Universe Baseline',
        '3': 'Geometric Optimizer',
        '4': 'Chemical Pivot',
        '5': 'Best vs Worst',
        '6': 'Hypothesis Validation',
        '7': 'Virtual Synthesis (2-Beam)'
    }
    return names.get(choice, 'Unknown')


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

    # 1. SELECT MODE (With or Without Agent 0)
    mode = get_mode_choice()
    use_agent0 = (mode == '1')
    
    # 2. GET INQUIRY (via Agent 0 or direct)
    problem_spec = None
    raw_user_query = ""
    
    if use_agent0:
        problem_spec, user_inquiry, raw_user_query = run_agent0_conversation()
        if user_inquiry is None:
            print("\n[System] Experiment cancelled.")
            return
    else:
        user_inquiry = get_direct_inquiry()
        raw_user_query = user_inquiry
    
    print(f"\n[System] Design Inquiry: {user_inquiry[:200]}...")
    
    # --- METRIC DETECTION ---
    # Priority-ordered keyword → (metric_column, display_name) mapping.
    # First match wins. hMOF-specific keywords checked BEFORE generic fallbacks.
    _KEYWORD_MAP = [
        # hMOF gas adsorption keywords (check first — more specific)
        (["ch4", "methane"],                    "ch4_uptake_35bar_298K",    "CH4 Uptake (35bar 298K)"),
        (["co2", "carbon dioxide"],             "co2_uptake_2_5bar_298K",  "CO2 Uptake (2.5bar 298K)"),
        (["xe/kr", "xe kr", "xenon", "krypton", "selectivity"],
                                                "xekr_selectivity_1bar",   "Xe/Kr Selectivity (1bar)"),
        (["100 bar", "100bar", "hmof"],         "h2_uptake_100bar_77K",    "H2 Uptake (100bar 77K, hMOF)"),
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

    for keywords, metric_col, display_name in _KEYWORD_MAP:
        if any(kw in inquiry_lower for kw in keywords):
            active_metric_column = metric_col
            active_metric_name = display_name
            found_metric = True
            print(f"[System] Target Metric Detected: {active_metric_name} ({metric_col})")
            break

    # Set config once here (single controlled mutation at startup, not at runtime)
    config.ACTIVE_METRIC_COLUMN = active_metric_column

    if not found_metric:
         print(f"[System] No specific metric detected. Defaulting to H2 Storage.")
    
    # 3. CREATE EXPERIMENT DIRECTORY
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    mode_suffix = "_agent0" if use_agent0 else "_direct"
    experiment_dir = os.path.join(EXPERIMENTS_DIR, f"exp_{timestamp}{mode_suffix}")
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"[System] Experiment directory: {experiment_dir}")
    
    # Save raw user input
    with open(os.path.join(experiment_dir, "raw_user_input.txt"), "w", encoding="utf-8") as f:
        f.write(raw_user_query)
    print(f"[System] Raw user input saved to raw_user_input.txt")
    
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

    agent1 = Agent1Handler()
    agent2 = Agent2Handler()
    
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
    memory = MemoryManager(experiment_dir, user_inquiry, model_name=ACTIVE_MODEL)
    logger = ExperimentLogger(experiment_dir)
    
    logger.log_model_info(ACTIVE_MODEL)
    logger.log_user_inquiry(user_inquiry)
    
    # Save Agent 0 output if used
    if use_agent0 and problem_spec:
        agent0_path = os.path.join(experiment_dir, "agent0_output.json")
        with open(agent0_path, 'w', encoding='utf-8') as f:
            json.dump(problem_spec, f, indent=2, ensure_ascii=False)
        print(f"[System] Saved Agent 0 output to {agent0_path}")
    
    # 5. ITERATION LOOP
    iteration = 0
    current_hypothesis = None
    current_feedback = ""
    scientific_journal = []
    
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
            # Subsequent iterations: refine based on feedback + Journal
            current_hypothesis = agent1.refine_hypothesis(current_feedback, scientific_journal)
        
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
        
        # ----- STEP E: FEEDBACK SELECTION (PI-DRIVEN) -----
        print("\n" + "="*50)
        print("SELECT FEEDBACK TYPE FOR AGENT 1")
        print("="*50)
        print("[1] 3-Beam Diagnostic     (Default, tests complete hypothesis against controls)")
        print("[2] Universe Baseline     (Samples across all DB, good if 0 hits)")
        print("[3] Geometric Optimizer   (Tests random vs constrained geometry)")
        print("[4] Chemical Pivot        (Tests random metal vs your geometry)")
        print("[5] Best vs Worst         (Stratified sampling to find patterns)")
        print("[6] Hypothesis Validation (Only tests the complete hypothesis block)")
        print("[7] Virtual Synthesis     (Lab synthesis simulation comparing chem vs chem+geo)")
        print("Type 'quit' to end the experiment.")
        print("-" * 50)
        
        feedback_type = 1
        while True:
            choice = input("Enter feedback type (1-7) or 'quit': ").strip().lower()
            if choice == 'quit':
                print("\n[System] Experiment manually terminated.")
                # We break out of the main loop
                break
            try:
                if 1 <= int(choice) <= 7:
                    feedback_type = int(choice)
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 7.")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 7, or 'quit'.")
                
        if choice == 'quit':
             break
        
        feedback_type_str = feedback_type_name(str(feedback_type))
        print(f"\n[System] Selected Feedback Type {feedback_type}: {feedback_type_str}")
        
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
        
        # ----- SCIENTIFIC JOURNAL UPDATE -----
        # Update Journal for NEXT iteration
        lesson = current_hypothesis.get('lesson_learnt', 'No lesson recorded.')
        nodes = current_hypothesis.get('node_composition', 'Unknown Nodes')
        
        journal_entry = f"Iteration {iteration}: [Nodes: {nodes}] | Lesson: {lesson}"
        scientific_journal.append(journal_entry)
        print(f"\n[Journal] Recorded: {journal_entry}")
        
        # Log to file
        logger.log_journal_update(journal_entry)
        
        # Save conversation history
        memory.save_conversation_history(agent1.get_conversation_history())
        
        print(f"\n[System] Iteration {iteration} complete.")
    
    # 6. EXPERIMENT END
    logger.log_experiment_end(iteration)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Mode: {'With Agent 0' if use_agent0 else 'Direct (No Agent 0)'}")
    print(f"Total Iterations: {iteration}")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_experiment()
