# AGENTS.md - LLM2POR Codebase Guide

## Project Overview

LLM2POR is an autonomous agent system that designs Metal-Organic Frameworks (MOFs) through iterative hypothesis generation, constraint extraction, and database-driven feedback. It uses LLMs (GPT/Gemini) to propose MOF designs and evaluates them against computational databases (QMOF, hMOF, PORMAKE).

**Entry Point**: `python run_experiment.py`

---

## Build / Test / Run Commands

### Running the Application

```bash
# Main entry point - interactive experiment runner
python run_experiment.py

# Run with specific environment
python -m venv llm2auto && source llm2auto/bin/activate
pip install -r requirements.txt
```

### Running Module Tests

Each core module contains a `test_<module>()` function that can be run directly:

```bash
# Test LLM client (requires API keys)
python -c "from core.llm_client import test_llm_client; test_llm_client()"

# Test Agent 1 handler
python -c "from core.agent1_handler import test_agent1; test_agent1()"
```

To run a single test, execute the module directly:

```bash
python core/llm_client.py
python core/agent1_handler.py
```

### Dependencies

Install via pip:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy==2.4.2
- openai==2.26.0
- pandas==3.0.1
- python-dotenv==1.1.0
- Requests==2.32.5
- scipy==1.17.1

### Environment Setup

Create `.env` file in project root:

```bash
# .env
LLM_PROVIDER=openai          # or "gemini"
OPENAI_API_KEY=sk-proj-...   # your OpenAI key
GEMINI_API_KEY=AIza...       # your Gemini key (if using Gemini)
```

---

## Code Style Guidelines

### Formatting & Layout

- **Line length**: Max 120 characters (soft limit)
- **Indentation**: 4 spaces (no tabs)
- **Section headers**: Use `# ====...====` pattern with title:
  ```python
  # =============================================================================
  # SECTION TITLE
  # =============================================================================
  ```
- **Module docstrings**: At top of file with purpose and description

### Imports

Group in order, separated by blank lines:

1. Standard library (`os`, `sys`, `json`, `datetime`)
2. Third-party packages (`pandas`, `numpy`, `requests`)
3. Local imports (relative imports from `core/`)

```python
import os
import sys
import json
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SOME_CONFIG
from core.llm_client import LLMClient
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `llm_client.py` |
| Classes | PascalCase | `Agent1Handler` |
| Functions | snake_case | `extract_json()` |
| Constants | UPPER_SNAKE | `LLM_MAX_OUTPUT_TOKENS` |
| Variables | snake_case | `user_inquiry` |
| Type aliases | PascalCase | `SensitivityDF = pd.DataFrame` |

### Type Hints

Use type hints for function signatures:

```python
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response text."""
    ...

def run_analysis(
    constraints: Dict[str, Any],
    matchmaker_results: Dict[str, Any],
    output_dir: str,
    run_id: str
) -> pd.DataFrame:
    ...
```

### Docstrings

Use Google-style or simple docstrings:

```python
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
```

### Error Handling

- **Never suppress errors silently**: Always log or print error messages
- **Use specific exception types**: Catch specific exceptions, not bare `Exception`
- **Fail fast with clear messages**: Provide actionable error information

```python
# Good
try:
    response = self.client.chat.completions.create(...)
except Exception as e:
    print(f"   [LLM ERROR] {e}")
    return None

# Bad - never do this
try:
    ...
except:
    pass  # Silent failure
```

### Class Structure

Follow this template for handler classes:

```python
class Agent1Handler:
    """
    Agent 1: Principal Investigator in Reticular Chemistry
    
    This agent generates MOF design hypotheses based on user inquiries
    and learns from feedback to refine its hypotheses over time.
    """
    
    def __init__(self):
        """Initialize Agent 1 with its system prompt."""
        ...
    
    def method_name(self, arg: type) -> return_type:
        """Method description."""
        ...
```

### Configuration

- All configuration in `config.py`
- Use UPPER_SNAKE_CASE for config variables
- Group related settings with section headers
- Use type hints on config functions

```python
# =============================================================================
# API CONFIGURATION
# =============================================================================

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
OPENAI_MODEL = "gpt-5.2"
GEMINI_MODEL = "gemini-3-flash-preview"
```

### JSON Handling

- Use `json.dumps()` with `indent=2` and `ensure_ascii=False` for human-readable output
- Use `json.loads()` with try/except for parsing
- The `LLMClient.extract_json()` method provides robust parsing with multiple fallback strategies

```python
with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

### Paths

- Use `os.path` for path operations (not pathlib for consistency)
- Use `os.path.dirname(os.path.abspath(__file__))` for reliable path resolution

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
```

---

## Project Structure

```
LLM2POR/
├── run_experiment.py          # Main entry point
├── config.py                  # All configuration
├── requirements.txt           # Dependencies
├── AGENTS.md                  # This file
├── .env                       # API keys (git-ignored)
├── core/                      # Runtime modules
│   ├── llm_client.py          # OpenAI/Gemini API wrapper
│   ├── agent0_handler.py      # Problem Consultant
│   ├── agent1_handler.py      # Hypothesis Generator
│   ├── agent2_handler.py      # Constraint Extractor
│   ├── matchmaker.py          # PORMAKE component matching
│   ├── qmof_matchmaker.py     # QMOF matching
│   ├── hmof_matchmaker.py     # hMOF matching
│   ├── sensitivity_analyzer.py # Performance evaluation
│   ├── feedback_generator.py  # Learning signal generation
│   ├── memory_manager.py      # Experiment state persistence
│   ├── name_resolver.py       # Building block ID resolver
│   └── constraint_utils.py    # Tag/ontology utilities
├── prompts/                   # LLM system prompts
│   ├── agent0_v3.md
│   ├── agent1_v2.2.9.md
│   └── agent2_v4.0.md
└── data/                      # Database files (Git LFS)
    ├── qmof_index_v2.json
    ├── hMOF/hmof_index.json
    └── ...
```

---

## Key Conventions

### Console Output

- Use brackets for system messages: `[System]`, `[Agent 1]`, `[Matchmaker]`
- Use `[LLM]` prefix for LLM-related messages
- Use `[ERROR]` and `[WARNING]` for error conditions

```python
print("\n[Agent 1] Generating initial hypothesis...")
print(f"   [LLM] Calling {self.model}...")
print(f"   [LLM ERROR] {e}")
```

### File Naming

- Output files: snake_case with descriptive suffixes
- Iteration directories: `iteration_{N}`
- Experiment directories: `exp_{YYYYMMDD}_{HHMM}_{mode}`

### Logging

- Use print statements for runtime feedback (no external logging library)
- Experiment logs saved to `experiments/` directory via `ExperimentLogger`

---

## Development Notes

- **No formal test suite**: Modules contain embedded `test_<module>()` functions
- **No linting config**: Follow existing code style in this file
- **Data files via Git LFS**: Run `git lfs pull` after cloning
- **Experiments directory**: Created at runtime, git-ignored
