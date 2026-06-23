# =============================================================================
# LLM4MOF Autonomous System - Memory Manager
# =============================================================================
# Manages conversation history and experiment context for transparency
# =============================================================================

import os
import json
import datetime
from typing import Dict, Any, List


class MemoryManager:
    """
    Manages the experiment's memory and context history.
    
    Stores:
    - Full conversation history from Agent 1 (for multi-turn)
    - Structured iteration data (hypotheses, feedback, results)
    - Exports to JSON for human analysis and debugging
    """
    
    def __init__(self, experiment_dir: str, user_inquiry: str, model_name: str = "unknown"):
        """
        Initialize memory manager for an experiment.
        
        Args:
            experiment_dir: Directory to save memory files
            user_inquiry: The user's initial design inquiry
            model_name: Name of the LLM model used
        """
        self.experiment_dir = experiment_dir
        self.experiment_id = os.path.basename(experiment_dir)
        
        # Initialize context structure
        self.context = {
            "experiment_id": self.experiment_id,
            "created_at": datetime.datetime.now().isoformat(),
            "llm_model": model_name,
            "user_inquiry": user_inquiry,
            "iterations": []
        }
        
        # Ensure directory exists
        os.makedirs(experiment_dir, exist_ok=True)
        
        print(f"[Memory] Initialized for experiment: {self.experiment_id}")
        print(f"[Memory] LLM Model: {model_name}")
    
    def add_iteration(self, iteration_num: int, hypothesis: Dict[str, Any],
                      constraints: Dict[str, Any], matchmaker_result: Dict[str, Any],
                      feedback_type: str, feedback_content: str,
                      sensitivity_summary: Dict[str, Any] = None):
        """
        Record data from a single iteration.
        
        Args:
            iteration_num: The iteration number
            hypothesis: Agent 1's hypothesis output
            constraints: Agent 2's constraints output
            matchmaker_result: Results from the matchmaker
            feedback_type: Type of feedback selected by user
            feedback_content: The actual feedback text sent to Agent 1
            sensitivity_summary: Optional summary of sensitivity analysis
        """
        iteration_data = {
            "iteration": iteration_num,
            "timestamp": datetime.datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "constraints": constraints,
            "matchmaker_counts": self._get_matchmaker_counts(matchmaker_result),
            "feedback_type": feedback_type,
            "feedback_content": feedback_content,
            "sensitivity_summary": sensitivity_summary
        }
        
        self.context["iterations"].append(iteration_data)
        
        # Save after each iteration
        self._save_context()
        
        print(f"[Memory] Iteration {iteration_num} recorded")
    
    @staticmethod
    def _get_matchmaker_counts(result: dict) -> dict:
        """Extract counts from matchmaker results (PORMAKE or QMOF)."""
        if result.get('qmof_mode'):
            return {
                "mode": "qmof",
                "qmof_matches": len(result.get('qmof_ids', [])),
                "diagnostics": result.get('diagnostics', {})
            }
        else:
            return {
                "mode": "pormake",
                "topologies": len(result.get('topology', [])),
                "nodes": len(result.get('node', [])),
                "edges": len(result.get('edge', [])),
                "diagnostics": result.get('diagnostics', {})
            }
    
    def save_conversation_history(self, history: List[Dict[str, str]]):
        """
        Save the full conversation history from Agent 1.
        
        Args:
            history: List of message dictionaries with 'role' and 'content'
        """
        history_path = os.path.join(self.experiment_dir, "conversation_history.json")
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"[Memory] Conversation history saved")
    
    def get_iteration_count(self) -> int:
        """Return the number of recorded iterations."""
        return len(self.context.get("iterations", []))

    def _save_context(self):
        """Save current context to JSON file."""
        context_path = os.path.join(self.experiment_dir, "context_history.json")
        
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(self.context, f, indent=2, ensure_ascii=False)


class ExperimentLogger:
    """
    Logs experiment progress to text files for human review.
    
    Creates:
    - experiment_log.txt: Main log with all iterations
    - Per-iteration logs with detailed outputs
    """
    
    def __init__(self, experiment_dir: str):
        """
        Initialize logger for an experiment.
        
        Args:
            experiment_dir: Directory to save log files
        """
        self.experiment_dir = experiment_dir
        self.log_path = os.path.join(experiment_dir, "experiment_log.txt")
        
        # Ensure directory exists
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Initialize log file
        self._write_log(f"{'='*60}\n")
        self._write_log(f"LLM4MOF EXPERIMENT LOG\n")
        self._write_log(f"Started: {datetime.datetime.now().isoformat()}\n")
        self._write_log(f"{'='*60}\n\n")
        
        print(f"[Logger] Initialized at: {self.log_path}")
    
    def log_model_info(self, model_name: str):
        """Log the LLM model used for this experiment."""
        self._write_log(f"LLM MODEL: {model_name}\n\n")

    def log_info(self, message: str):
        """Log a generic information message."""
        self._write_log(f"[Info] {message}\n")
    
    def _write_log(self, text: str, also_print: bool = False):
        """Append text to the main log file."""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(text)
        
        if also_print:
            print(text, end='')
    
    def log_user_inquiry(self, inquiry: str):
        """Log the user's initial inquiry."""
        self._write_log(f"USER INQUIRY:\n{inquiry}\n\n")
    
    def log_iteration_start(self, iteration_num: int):
        """Log the start of an iteration."""
        self._write_log(f"\n{'='*60}\n")
        self._write_log(f"ITERATION {iteration_num}\n")
        self._write_log(f"Time: {datetime.datetime.now().isoformat()}\n")
        self._write_log(f"{'='*60}\n\n")
    
    def log_hypothesis(self, hypothesis: Dict[str, Any]):
        """Log Agent 1's hypothesis."""
        self._write_log(f"--- AGENT 1 HYPOTHESIS ---\n")
        self._write_log(json.dumps(hypothesis, indent=2) + "\n\n")
    
    def log_constraints(self, constraints: Dict[str, Any]):
        """Log Agent 2's constraints."""
        self._write_log(f"--- AGENT 2 CONSTRAINTS ---\n")
        self._write_log(json.dumps(constraints, indent=2) + "\n\n")

    
    def log_matchmaker_results(self, results: Dict[str, Any]):
        """Log matchmaker results. PORMAKE reports an assembled N x L x T design space; the
        direct-match databases (QMOF / hMOF / CoRE-MOF) report a flat count of matched MOF ids."""
        self._write_log(f"--- MATCHMAKER RESULTS ---\n")
        if isinstance(results, dict):
            # (display label, mode flag, ids key) for the direct-database-matching modes
            _direct = (("QMOF", "qmof_mode", "qmof_ids"),
                       ("hMOF", "hmof_mode", "hmof_ids"))
            hit = next(((lbl, ids_key) for lbl, flag, ids_key in _direct if results.get(flag)), None)
            if hit:
                label, ids_key = hit
                self._write_log(f"Mode:        {label} (Direct Matching)\n")
                self._write_log(f"Matches:     {len(results.get(ids_key, []))}\n")
            else:
                t = len(results.get('topology', []))
                n = len(results.get('node', []))
                e = len(results.get('edge', []))
                self._write_log(f"Topologies:  {t}\n")
                self._write_log(f"Nodes:       {n}\n")
                self._write_log(f"Edges:       {e}\n")
                self._write_log(f"Total Space: {t * n * e:,} combinations\n")
        else:
            self._write_log(f"Error: {results}\n")
        self._write_log("\n")
    
    def log_sensitivity_report(self, report_text: str):
        """Log the sensitivity analysis report."""
        self._write_log(f"--- SENSITIVITY ANALYSIS (Human Only) ---\n")
        self._write_log(report_text + "\n\n")
    
    def log_feedback_selection(self, feedback_type: str, feedback_content: str):
        """Log the feedback selected and sent to Agent 1."""
        self._write_log(f"--- FEEDBACK SELECTED ---\n")
        self._write_log(f"Type: {feedback_type}\n")
        self._write_log(f"Content:\n{feedback_content}\n\n")
    
    def log_experiment_end(self, total_iterations: int):
        """Log the end of the experiment."""
        self._write_log(f"\n{'='*60}\n")
        self._write_log(f"EXPERIMENT COMPLETE\n")
        self._write_log(f"Total Iterations: {total_iterations}\n")
        self._write_log(f"Ended: {datetime.datetime.now().isoformat()}\n")
        self._write_log(f"{'='*60}\n")


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_memory_and_logger():
    """Test memory manager and logger."""
    
    print("\n" + "="*60)
    print("MEMORY MANAGER & LOGGER TEST")
    print("="*60 + "\n")
    
    import tempfile
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix="llm4mof_test_")
    print(f"Test directory: {test_dir}")
    
    # Initialize
    inquiry = "Design a MOF for H2 storage"
    memory = MemoryManager(test_dir, inquiry)
    logger = ExperimentLogger(test_dir)
    
    # Log user inquiry
    logger.log_user_inquiry(inquiry)
    
    # Simulate an iteration
    sample_hypothesis = {"target_application": "H2 storage", "database_constraints": {"metal": "Zn"}}
    sample_constraints = {"node_query": {"metal_symbol": "Zn", "connectivity": 4}}
    sample_matchmaker = {"topology": ["fcu", "pcu"], "node": ["N1"], "edge": ["E1", "E2"]}
    
    logger.log_iteration_start(1)
    logger.log_hypothesis(sample_hypothesis)
    logger.log_constraints(sample_constraints)
    logger.log_matchmaker_results(sample_matchmaker)
    
    memory.add_iteration(
        iteration_num=1,
        hypothesis=sample_hypothesis,
        constraints=sample_constraints,
        matchmaker_result=sample_matchmaker,
        feedback_type="4-Beam Diagnostic",
        feedback_content="Sample feedback..."
    )
    
    # Verify files created
    context_file = os.path.join(test_dir, "context_history.json")
    log_file = os.path.join(test_dir, "experiment_log.txt")
    
    print(f"\n--- Verification ---")
    print(f"Context file exists: {os.path.exists(context_file)}")
    print(f"Log file exists: {os.path.exists(log_file)}")
    print(f"Iteration count: {memory.get_iteration_count()}")
    
    if os.path.exists(context_file) and os.path.exists(log_file):
        print("✓ MEMORY & LOGGER TEST PASSED")
    else:
        print("✗ TEST FAILED")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_memory_and_logger()
