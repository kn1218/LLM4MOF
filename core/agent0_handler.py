# =============================================================================
# LLM2POR Autonomous System - Agent 0 Handler
# =============================================================================
# Problem Consultant: Multi-turn conversation to create structured problem spec
# =============================================================================

import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AGENT0_PROMPT_PATH, CAPABILITY_MANIFEST, AGENT0_MAX_TURNS, AGENT0_SKIP_COMMANDS
from core.llm_client import LLMClient, load_prompt


class Agent0Handler:
    """
    Agent 0: Problem Consultant
    
    This agent interviews the user through multi-turn conversation to create
    a structured problem specification JSON that will be consumed by Agent 1.
    
    Key responsibilities:
    - Ask clarifying questions about operating conditions, goals, constraints
    - Disclose assumptions before finalizing
    - Output structured JSON problem specification
    """
    
    def __init__(self):
        """Initialize Agent 0 with its system prompt."""
        # Load system prompt from file
        self.system_prompt = load_prompt(AGENT0_PROMPT_PATH)
        
        # Inject capability manifest into system prompt
        manifest_text = f"\n\nCAPABILITY MANIFEST:\n{json.dumps(CAPABILITY_MANIFEST, indent=2)}"
        self.system_prompt += manifest_text
        
        # Initialize LLM client with multi-turn enabled
        self.client = LLMClient(self.system_prompt, multi_turn=True)
        
        # Track conversation state
        self.conversation_complete = False
        self.problem_spec = None
        
        print("[Agent 0] Initialized - Problem Consultant (Multi-turn mode)")
    
    def start_conversation(self, initial_query: str) -> str:
        """
        Start the problem formulation conversation.
        
        Args:
            initial_query: The user's initial (possibly vague) query
            
        Returns:
            Agent 0's first response (likely clarifying questions)
        """
        print("\n[Agent 0] Starting problem formulation...")
        print(f"   Initial Query: {initial_query}")
        
        # Send initial query to Agent 0
        response = self.client.send_message(initial_query)
        
        if not response:
            return "Error: No response from Agent 0"
        
        # Check if Agent 0 already has enough info and outputs JSON
        if self._check_for_json_output(response):
            self.conversation_complete = True
            return response
        
        return response
    
    def continue_conversation(self, user_response: str) -> str:
        """
        Continue the conversation with user's response to questions.
        
        Args:
            user_response: User's answer to Agent 0's questions
            
        Returns:
            Agent 0's next response or final JSON
        """
        # Handle skip/proceed commands
        if user_response.lower().strip() in AGENT0_SKIP_COMMANDS:
            user_response = "Please proceed with the information you have. List any assumptions you're making."
        
        response = self.client.send_message(user_response)
        
        if not response:
            return "Error: No response from Agent 0"
        
        # Check if Agent 0 outputs final JSON
        if self._check_for_json_output(response):
            self.conversation_complete = True
        
        return response
    
    def _check_for_json_output(self, response: str) -> bool:
        """
        Check if the response contains the final JSON output.
        
        Returns:
            True if valid problem spec JSON found
        """
        json_data = LLMClient.extract_json(response)
        
        if json_data and 'problem_specification' in json_data:
            self.problem_spec = json_data
            return True
        
        return False
    
    def get_problem_spec(self) -> Optional[Dict[str, Any]]:
        """
        Get the final problem specification.
        
        Returns:
            Problem specification JSON or None if not complete
        """
        return self.problem_spec
    
    def get_user_inquiry_text(self) -> str:
        """
        Convert problem spec to formatted text for Agent 1.
        
        This provides a rich, structured inquiry instead of raw user text.
        """
        if not self.problem_spec:
            return ""
        
        spec = self.problem_spec
        ps = spec.get('problem_specification', {})
        constraints = spec.get('constraints', {})
        goals = spec.get('optimization_goals', {})
        flags = spec.get('diagnostic_flags', {})
        
        # Build enriched inquiry text
        text_parts = []
        
        # Domain and targets
        domain = ps.get('domain_category', 'Unknown')
        targets = ps.get('target_molecules', [])
        text_parts.append(f"Domain: {domain}")
        text_parts.append(f"Target molecules: {', '.join(targets) if targets else 'Not specified'}")
        
        # Operating conditions
        op = ps.get('operating_conditions', {})
        if op:
            cond_str = f"Operating conditions: {op.get('temperature_range', 'N/A')} at {op.get('pressure_range', 'N/A')}, {op.get('phase', 'N/A')} phase"
            text_parts.append(cond_str)
        
        # Optimization goals
        if goals:
            text_parts.append(f"Primary optimization goal: {goals.get('primary_metric', 'N/A')}")
            if goals.get('secondary_metric'):
                text_parts.append(f"Secondary goal: {goals.get('secondary_metric')}")
        
        # Constraints
        if constraints.get('must_have'):
            text_parts.append(f"Must have: {', '.join(constraints['must_have'])}")
        if constraints.get('must_avoid'):
            text_parts.append(f"Must avoid: {', '.join(constraints['must_avoid'])}")
        
        # Stability
        stability = constraints.get('stability_requirements', {})
        for stab_type, status in stability.items():
            if status == 'required':
                text_parts.append(f"Requirement: {stab_type} is REQUIRED")
            elif status == 'irrelevant':
                 text_parts.append(f"Constraint: {stab_type} is IRRELEVANT (Do not prioritize)")

        
        # Assumptions made
        if flags.get('assumptions_made'):
            text_parts.append(f"Note: The following assumptions were made: {', '.join(flags['assumptions_made'])}")
        
        return "\n".join(text_parts)
    
    def is_complete(self) -> bool:
        """Check if the conversation is complete."""
        return self.conversation_complete


def run_agent0_interview() -> Optional[Dict[str, Any]]:
    """
    Run a complete Agent 0 interview session.
    
    Returns:
        Final problem specification JSON or None if cancelled
    """
    print("\n" + "="*60)
    print("MOF DESIGN PROBLEM FORMULATION")
    print("="*60)
    print("Agent 0 will help clarify your requirements.")
    print("Type 'proceed' anytime to skip remaining questions.")
    print("Type 'quit' to cancel.")
    print("="*60 + "\n")
    
    # Get initial query
    initial_query = input("What MOF do you want to design? → ").strip()
    
    if initial_query.lower() == 'quit':
        return None
    
    # Initialize Agent 0
    agent0 = Agent0Handler()
    
    # Start conversation
    response = agent0.start_conversation(initial_query)
    print(f"\n[Agent 0]: {response}\n")
    
    # Continue until complete or user quits
    max_turns = AGENT0_MAX_TURNS
    turn = 0
    
    while not agent0.is_complete() and turn < max_turns:
        user_input = input("Your response → ").strip()
        
        if user_input.lower() == 'quit':
            print("\n[System] Interview cancelled.")
            return None
        
        response = agent0.continue_conversation(user_input)
        print(f"\n[Agent 0]: {response}\n")
        turn += 1
    
    if agent0.is_complete():
        print("\n" + "="*60)
        print("PROBLEM SPECIFICATION COMPLETE")
        print("="*60)
        print(json.dumps(agent0.get_problem_spec(), indent=2))
        return agent0.get_problem_spec()
    else:
        print("\n[System] Max turns reached. Using partial specification.")
        return agent0.get_problem_spec()


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_agent0():
    """Test Agent 0 with a simulated conversation."""
    
    print("\n" + "="*60)
    print("AGENT 0 HANDLER TEST")
    print("="*60 + "\n")
    
    # Initialize agent
    agent0 = Agent0Handler()
    
    # Test with a vague query
    test_query = "I need a MOF for hydrogen"
    
    print(f"Initial query: {test_query}")
    response = agent0.start_conversation(test_query)
    print(f"\n[Agent 0 Response]:\n{response[:500]}...")
    
    # Simulate user response
    if not agent0.is_complete():
        print("\n[Simulating user response]")
        response = agent0.continue_conversation(
            "77K temperature, high pressure around 100 bar, for storage application"
        )
        print(f"\n[Agent 0 Response]:\n{response[:500]}...")
    
    # Check if complete
    if not agent0.is_complete():
        print("\n[Simulating proceed command]")
        response = agent0.continue_conversation("proceed")
        print(f"\n[Agent 0 Response]:\n{response[:800]}...")
    
    if agent0.is_complete():
        print("\n--- Validation ---")
        spec = agent0.get_problem_spec()
        if spec and 'problem_specification' in spec:
            print("✓ AGENT 0 TEST PASSED - Problem specification generated")
            print(f"\nEnriched inquiry for Agent 1:\n{agent0.get_user_inquiry_text()}")
        else:
            print("✗ Missing problem_specification field")
    else:
        print("✗ AGENT 0 TEST FAILED - Conversation did not complete")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Run interactive interview
    result = run_agent0_interview()
    if result:
        print("\nFinal specification saved.")
