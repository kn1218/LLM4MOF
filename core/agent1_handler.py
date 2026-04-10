# =============================================================================
# LLM2POR Autonomous System - Agent 1 Handler
# =============================================================================
# Hypothesis Generator with multi-turn memory
# =============================================================================

import os
import sys
import datetime
from typing import Optional, Dict, Any

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_agent1_prompt_path
from core.llm_client import LLMClient, load_prompt


class Agent1Handler:
    """
    Agent 1: Principal Investigator in Reticular Chemistry
    
    This agent generates MOF design hypotheses based on user inquiries
    and learns from feedback to refine its hypotheses over time.
    
    Uses multi-turn conversation to maintain memory of past hypotheses
    and feedback across iterations.
    """
    
    def __init__(self):
        """Initialize Agent 1 with its system prompt."""
        # Load system prompt from file
        self.system_prompt = load_prompt(get_agent1_prompt_path())
        
        # Initialize LLM client with multi-turn enabled
        self.client = LLMClient(self.system_prompt, multi_turn=True)
        
        print("[Agent 1] Initialized - Hypothesis Generator (Multi-turn mode)")

    def generate_initial_hypothesis(self, user_inquiry: str) -> Optional[Dict[str, Any]]:
        """
        Generate the first hypothesis based on user inquiry.

        Args:
            user_inquiry: The user's research goal (e.g., "Design MOF for H2 storage")

        Returns:
            Parsed JSON hypothesis or None if failed
        """
        print("\n[Agent 1] Generating initial hypothesis...")
        print(f"   User Inquiry: {user_inquiry}")

        # Chemistry-first guidance: prevent over-constraining on iteration 1
        first_iter_guidance = (
            "\n\nFIRST-ITERATION STRATEGY: You have no feedback yet. "
            "Focus on CHEMISTRY only - specify metals and linker functional groups "
            "based on your domain knowledge. Leave geometry_filter EMPTY or specify "
            "at most 1-2 key descriptors. Do NOT constrain all geometry parameters "
            "simultaneously - the database is finite and each constraint removes ~50%% "
            "of candidates. You will refine geometry in iteration 2+ based on the "
            "4-beam diagnostic data."
        )
        response = self.client.send_message(user_inquiry + first_iter_guidance)
        
        if not response:
            print("[Agent 1] ERROR: No response received")
            return None
        
        # Extract JSON from response
        hypothesis = LLMClient.extract_json(response)
        
        if hypothesis:
            self._print_hypothesis(hypothesis)
            return hypothesis
        else:
            print("[Agent 1] ERROR: Could not parse hypothesis JSON")
            print(f"   Raw response: {response[:500]}...")
            self._dump_raw_response(response, "initial")
            return None
    
    def refine_hypothesis(self, feedback: str) -> Optional[Dict[str, Any]]:
        """
        Generate a refined hypothesis based on feedback.

        The multi-turn conversation maintains context of previous
        hypotheses, so the agent can learn and improve.

        Args:
            feedback: Feedback text from the simulation results

        Returns:
            Parsed JSON hypothesis or None if failed
        """
        print("\n[Agent 1] Processing feedback and refining hypothesis...")
        
        # Construct feedback message
        feedback_message = f"""
Based on the experimental laboratory feedback below, please analyze your previous hypothesis.

Read the provided report carefully to understand the physical and chemical realities of your chosen components. 
Pay close attention to the "STATUS" and explicitly follow any "INSTRUCTION" or "HYPOTHESIS EVALUATION TASK" provided in the text.

IMPORTANT: The feedback contains 4 diagnostic beams. Each beam is clearly labeled with what it controls for.
Read the beam descriptions carefully — they tell you exactly what variable each beam isolates.
Compare adjacent beams to diagnose which part of your hypothesis (metals, linkers, geometry, or overall chemistry) is helping or hurting.
Each beam includes a "Chemistry Profile" and "Pattern Summary". Use these to identify WHICH chemical properties correlate with high performance.

Use this empirical data to deduce what worked, what failed, and how to self-correct.
State whether you are executing an 'Exploitation Phase' or an 'Exploration Phase' in your reasoning.
Then, propose an improved hypothesis. Maintain the exact same JSON output format.

{feedback}
"""
        
        response = self.client.send_message(feedback_message)
        
        if not response:
            print("[Agent 1] ERROR: No response received")
            return None
        
        # Extract JSON from response
        hypothesis = LLMClient.extract_json(response)
        
        if hypothesis:
            self._print_hypothesis(hypothesis)
            return hypothesis
        else:
            print("[Agent 1] ERROR: Could not parse refined hypothesis JSON")
            print(f"   Raw response: {response[:500]}...")
            self._dump_raw_response(response, "refined")
            return None

    def _dump_raw_response(self, response: str, label: str):
        """Save full raw LLM response to a debug file for post-mortem analysis."""
        try:
            from config import EXPERIMENTS_DIR
            debug_dir = os.path.join(EXPERIMENTS_DIR, "_debug")
            os.makedirs(debug_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = os.path.join(debug_dir, f"agent1_raw_{label}_{ts}.txt")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Agent 1 Raw LLM Response ({label}) ===\n")
                f.write(f"Timestamp: {ts}\n")
                f.write(f"Response length: {len(response)} chars\n")
                f.write(f"{'='*60}\n\n")
                f.write(response)
            print(f"   [DEBUG] Full raw response saved to: {debug_path}")
        except Exception as e:
            print(f"   [DEBUG] Could not save raw response: {e}")

    def _print_hypothesis(self, hypothesis: Dict[str, Any]):
        """Pretty print the hypothesis for CLI display."""
        print("\n" + "="*50)
        print("AGENT 1 HYPOTHESIS")
        print("="*50)
        
        # Extract key fields
        target = hypothesis.get('target_application', 'N/A')
        mechanism = hypothesis.get('hypothesis_mechanism', 'N/A')
        
        print(f"Target: {target}")
        print(f"Mechanism: {mechanism[:100]}..." if len(str(mechanism)) > 100 else f"Mechanism: {mechanism}")
        
        # Node and linker composition
        node = hypothesis.get('node_composition', 'N/A')
        linker = hypothesis.get('linker_composition', 'N/A')
        geometry = hypothesis.get('ideal_pore_geometry', 'N/A')
        
        print("-" * 20)
        print(f"Node: {node}")
        print(f"Linker: {linker}")
        print(f"Geometry: {geometry}")
        
        # PI Logic (Meta-Cognition)
        if 'meta_cognition' in hypothesis:
            print("-" * 20)
            print(f"Reasoning: {hypothesis['meta_cognition'].get('reasoning', 'No reasoning provided')}")

        lesson = hypothesis.get('lesson_learnt')
        if lesson:
            print("-" * 20)
            print(f"Lesson Learnt: {lesson}")
        
        print("="*50)
    
    def get_conversation_history(self):
        """Return the full conversation history for logging."""
        return self.client.get_conversation_history()

    def set_conversation_history(self, history):
        """Restore conversation history from checkpoint."""
        self.client.set_conversation_history(history)



# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_agent1():
    """Test Agent 1 with a sample inquiry."""
    
    print("\n" + "="*60)
    print("AGENT 1 HANDLER TEST")
    print("="*60 + "\n")
    
    # Initialize agent
    agent1 = Agent1Handler()
    
    # Test initial hypothesis
    test_inquiry = "Design a MOF for high capacity Hydrogen storage at 77K"
    hypothesis = agent1.generate_initial_hypothesis(test_inquiry)
    
    if hypothesis:
        print("\n--- Validation ---")
        # Check required fields
        required = ['target_application', 'node_composition', 'ideal_pore_geometry']
        missing = [f for f in required if f not in hypothesis]
        
        if not missing:
            if 'node_composition' in hypothesis:
                print("✓ AGENT 1 TEST PASSED - Hypothesis generated with required fields")
            else:
                print("✗ Missing node_composition field")
        else:
            print(f"✗ Missing required fields: {missing}")
    else:
        print("✗ AGENT 1 TEST FAILED - No hypothesis generated")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_agent1()
