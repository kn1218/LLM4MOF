# =============================================================================
# LLM2POR Autonomous System - LLM Client
# =============================================================================
# Unified LLM Client supporting both OpenAI and Gemini APIs
# =============================================================================

import os
import sys
import re
import json
import requests
from typing import Optional, List, Dict, Any

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLM_PROVIDER, ACTIVE_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    LLM_MAX_OUTPUT_TOKENS, LLM_MAX_RETRIES, LLM_REQUEST_TIMEOUT
)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Run: pip install openai")


def load_prompt(path: str) -> str:
    """
    Load a prompt file with error handling.

    Args:
        path: Absolute path to the prompt markdown file

    Returns:
        The prompt text content

    Raises:
        FileNotFoundError: If the prompt file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class LLMClient:
    """
    Unified LLM Client supporting both OpenAI and Gemini APIs.
    
    For Agent 1: Maintains conversation history (multi-turn mode)
    For Agent 2: Stateless single calls
    """
    
    def __init__(self, system_prompt: str, multi_turn: bool = False):
        """
        Initialize the LLM client based on configured provider.
        
        Args:
            system_prompt: The system prompt defining the agent's role
            multi_turn: If True, maintain conversation history (for Agent 1)
        """
        self.provider = LLM_PROVIDER
        self.system_prompt = system_prompt
        self.multi_turn = multi_turn
        
        if self.provider == "openai":
            self.model = OPENAI_MODEL
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI package not installed. Run: pip install openai")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:  # gemini
            self.model = GEMINI_MODEL
            self.api_key = GEMINI_API_KEY
            self.client = None  # REST API, no client needed
        
        # Initialize conversation history with system prompt
        self.conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
    
    def send_message(self, user_message: str, temperature: float = 0.7) -> Optional[str]:
        """
        Send a message and get a response from the configured LLM provider.
        """
        if self.provider == "openai":
            return self._send_openai(user_message, temperature)
        else:
            return self._send_gemini(user_message, temperature)
    
    def _send_openai(self, user_message: str, temperature: float) -> Optional[str]:
        """Send message via OpenAI API."""
        try:
            if self.multi_turn:
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                messages = self.conversation_history
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ]
            
            print(f"   [LLM] Calling {self.model}...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=LLM_MAX_OUTPUT_TOKENS,
                    response_format={"type": "json_object"}
                )
            except Exception as json_mode_err:
                # Fallback: some models don't support response_format
                print(f"   [LLM] JSON mode not supported ({json_mode_err}), retrying without it...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=LLM_MAX_OUTPUT_TOKENS
                )
            
            assistant_message = response.choices[0].message.content

            # Check for truncation (response hit token limit)
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                print(f"   [LLM WARNING] Response was TRUNCATED (hit {LLM_MAX_OUTPUT_TOKENS} token limit).")
                print(f"   [LLM WARNING] JSON output is likely incomplete. Consider increasing LLM_MAX_OUTPUT_TOKENS in config.py.")

            if self.multi_turn:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
            
            return assistant_message
            
        except Exception as e:
            print(f"   [LLM ERROR] {e}")
            return None
    
    def _send_gemini(self, user_message: str, temperature: float) -> Optional[str]:
        """Send message via Gemini REST API."""
        max_retries = LLM_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                # Build conversation for Gemini format
                if self.multi_turn and attempt == 0:  # Only add on first attempt
                    self.conversation_history.append({
                        "role": "user",
                        "content": user_message
                    })
                
                # Construct contents for Gemini API
                contents = []
                
                # Add system instruction as first user message context
                # Gemini uses a different format - system prompt goes in systemInstruction
                for msg in (self.conversation_history if self.multi_turn else 
                           [{"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_message}]):
                    if msg["role"] == "system":
                        continue  # Handled separately
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append({
                        "role": role,
                        "parts": [{"text": msg["content"]}]
                    })
                
                # Gemini API endpoint
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                
                payload = {
                    "contents": contents,
                    "systemInstruction": {
                        "parts": [{"text": self.system_prompt}]
                    },
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": LLM_MAX_OUTPUT_TOKENS,
                        "responseMimeType": "application/json"
                    }
                }
                
                print(f"   [LLM] Calling {self.model}...")
                response = requests.post(url, json=payload, timeout=LLM_REQUEST_TIMEOUT)
                
                if response.status_code != 200:
                    print(f"   [LLM ERROR] Status {response.status_code}: {response.text[:500]}")
                    return None
                
                result = response.json()
                
                # Extract text from response
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    
                    # Handle empty content (thinking models sometimes return empty content)
                    if "content" not in candidate or not candidate["content"]:
                        if attempt < max_retries - 1:
                            print(f"   [LLM] Empty content received, retrying ({attempt + 1}/{max_retries})...")
                            continue
                        else:
                            print(f"   [LLM ERROR] Empty content after {max_retries} attempts")
                            return None
                    
                    if "parts" in candidate["content"]:
                        assistant_message = ""
                        for part in candidate["content"]["parts"]:
                            if "text" in part:
                                assistant_message += part["text"]
                        
                        if assistant_message:
                            # SP-3.03: Check for Gemini truncation
                            finish_reason = candidate.get('finishReason', '')
                            if finish_reason == 'MAX_TOKENS':
                                print(f"   [LLM WARNING] Gemini response TRUNCATED (finishReason={finish_reason}).")
                                print(f"   [LLM WARNING] JSON output is likely incomplete. Consider increasing LLM_MAX_OUTPUT_TOKENS in config.py.")
                            elif finish_reason and finish_reason != 'STOP':
                                print(f"   [LLM] Gemini finishReason: {finish_reason}")
                            
                            if self.multi_turn:
                                self.conversation_history.append({
                                    "role": "assistant",
                                    "content": assistant_message
                                })
                            return assistant_message
                        else:
                            if attempt < max_retries - 1:
                                print(f"   [LLM] No text in response, retrying ({attempt + 1}/{max_retries})...")
                                continue
                
                print(f"   [LLM ERROR] Unexpected response format: {result}")
                return None
                
            except Exception as e:
                print(f"   [LLM ERROR] {e}")
                return None
        
        return None
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the full conversation history."""
        return self.conversation_history.copy()
    
    def set_conversation_history(self, history: List[Dict[str, str]]) -> None:
        """Restore conversation history (e.g., from checkpoint)."""
        self.conversation_history = history.copy()

    def clear_history(self):
        """Clear conversation history, keeping only system prompt."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response text.
        
        Tries multiple strategies:
        1. Parse entire text as JSON
        2. Find JSON block in markdown code fence
        3. Find JSON-like structure with regex
        """
        if not text:
            return None

        last_error = None  # Track last JSONDecodeError for diagnostics
            
        # Strategy 1: Try parsing entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_error = e
            pass
        
        # Strategy 2: Find JSON in markdown code fence
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError as e:
                last_error = e
                continue
        
        # Strategy 3: Find JSON-like structure with braces
        brace_pattern = r'\{[\s\S]*\}'
        matches = re.findall(brace_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError as e:
                last_error = e
                continue
        
        # Strategy 4: Attempt to repair truncated JSON (missing closing braces/brackets)
        stripped = text.strip()
        if stripped.startswith('{'):
            # Count unmatched braces/brackets
            open_braces = stripped.count('{') - stripped.count('}')
            open_brackets = stripped.count('[') - stripped.count(']')

            if open_braces > 0 or open_brackets > 0:
                # Truncate at last complete value (before a trailing comma or incomplete string)
                repaired = stripped.rstrip(',\n\r\t ')
                # Close any open strings (heuristic: if odd number of unescaped quotes)
                if repaired.count('"') % 2 != 0:
                    repaired += '"'
                # Close brackets then braces
                repaired += ']' * max(0, open_brackets) + '}' * max(0, open_braces)
                try:
                    result = json.loads(repaired)
                    print("   [JSON EXTRACTION] Repaired truncated JSON (missing closing braces)")
                    return result
                except json.JSONDecodeError as e:
                    last_error = e
                    pass

        # All strategies failed — print diagnostic details
        print("   [JSON EXTRACTION] Failed to extract valid JSON from response")
        if last_error:
            print(f"   [JSON EXTRACTION] Last parse error: {last_error}")
            print(f"   [JSON EXTRACTION] Error at position {last_error.pos}, line {last_error.lineno}, col {last_error.colno}")
        print(f"   [JSON EXTRACTION] Response length: {len(text)} chars, starts with: {repr(text[:80])}")
        return None




# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_llm_client():
    """Test the LLM client with a simple prompt."""
    
    print("\n" + "="*60)
    print("LLM CLIENT MODULE TEST")
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Model: {ACTIVE_MODEL}")
    print("="*60 + "\n")
    
    # Test with a simple prompt
    system_prompt = "You are a helpful assistant that responds in JSON format."
    
    try:
        client = LLMClient(system_prompt, multi_turn=False)
        
        # Test basic response
        print("--- Testing Basic Response ---")
        response = client.send_message('Say "Hello" and include a "status": "ok" field in JSON.')
        
        if response:
            print(f"Response received ({len(response)} chars)")
            print(f"First 200 chars: {response[:200]}...")
            
            # Test JSON extraction
            print("\n--- Testing JSON Extraction ---")
            extracted = LLMClient.extract_json(response)
            if extracted:
                print(f"JSON extracted successfully: {extracted}")
                print("✓ LLM CLIENT TEST PASSED")
            else:
                print("✗ JSON extraction failed")
        else:
            print("✗ No response received")
            
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_llm_client()
