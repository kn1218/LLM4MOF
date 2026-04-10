# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM2POR** is an autonomous Metal-Organic Framework (MOF) designer. LLMs (GPT/Gemini) iteratively propose MOF designs evaluated against real computational databases: QMOF (20K MOFs, electronic properties), hMOF (51K hypothetical MOFs, gas adsorption), and PORMAKE (component assembly from building blocks).

See `AGENTS.md` for code style guidelines, full project structure, and optional simulation pipeline commands.

## Commands

```bash
# Setup
python -m venv llm2auto && source llm2auto/bin/activate
pip install -r requirements.txt
git lfs pull                  # Download large database files (~174 MB)

# Run experiment
python run_experiment.py

# Run a single module test (no pytest — each module has embedded test_<module>())
python core/llm_client.py
python core/matchmaker.py
python core/feedback_generator.py
python core/sensitivity_analyzer.py
# etc. (see AGENTS.md for full list)
```

### Environment setup (`.env`)
```
LLM_PROVIDER=openai           # or "gemini"
OPENAI_API_KEY=sk-proj-...
# GEMINI_API_KEY=...          # if using Gemini
```

## Architecture

### The Autonomous Loop

Each experiment iteration runs these steps in sequence:

```
User Inquiry
    ↓
[Agent 0] Problem Consultant (optional interview, agent0_handler.py)
    ↓
[Agent 1] Hypothesis Generator — multi-turn, stateful, scientific journal (agent1_handler.py)
    ↓  JSON hypothesis
[Agent 2] Constraint Extractor — stateless translation to DB search format (agent2_handler.py)
    ↓  quantitative constraints
[Matchmaker] Component/MOF Discovery — auto-selects database based on inquiry mode
    ↓  ranked candidates
[Sensitivity Analyzer] Enrichment Factor + p-value evaluation (sensitivity_analyzer.py)
    ↓  statistical metrics
[Feedback Generator] 6 feedback types → Agent 1 refines hypothesis (feedback_generator.py)
```

### Database Modes (auto-detected from inquiry keywords)

| Mode | Matchmaker | Database | Target Property |
|------|------------|----------|-----------------|
| PORMAKE | `matchmaker.py` | 50K+ building blocks | H2 storage, custom |
| QMOF | `qmof_matchmaker.py` | 20,373 MOFs | Band gap |
| hMOF | `hmof_matchmaker.py` | 51,163 MOFs | CH4, CO2, Xe/Kr, H2 adsorption |

Mode-detection functions: `is_qmof_mode()`, `is_hmof_mode()` in `config.py`.

### Key Design Decisions

- **Agent 1 is stateful**: Maintains multi-turn conversation + explicit scientific journal of past iterations. Must list all constraints explicitly every iteration (no memory assumptions).
- **Agent 2 is stateless**: Single-shot translation; validated against canonical vocabulary from `data/unified_ontology.json` (80+ approved functional group tags).
- **Feedback is hidden from agents**: Enrichment Factor and p-values are for human analysis only; Agent 1 only receives structured qualitative/quantitative feedback.
- **No pytest**: Tests are `if __name__ == "__main__"` blocks executed by running the module directly.
- **All config in `config.py`**: Database paths, API settings, metric registry, display settings — all centralized here.

### Critical Files

| File | Role |
|------|------|
| `run_experiment.py` | Main orchestrator — experiment loop, user interaction |
| `config.py` | All configuration, paths, capability manifest, metric registry |
| `core/llm_client.py` | Unified OpenAI/Gemini wrapper; use `LLMClient.extract_json()` for robust JSON parsing |
| `prompts/agent1_v*.md` | Agent 1 system prompt — encodes the core scientific reasoning strategy |
| `prompts/agent2_v*.md` | Agent 2 system prompt — enforces canonical vocabulary |
| `data/unified_ontology.json` | Source of truth for functional group vocabulary |

### Output Structure

Experiments write to `experiments/exp_<timestamp>/`:
- `agent1_output.json`, `agent2_output.json` — per-iteration outputs
- `matchmaker_results.json` — candidate MOFs
- `sensitivity_analysis.csv` — enrichment metrics
- `feedback.json` — feedback sent to Agent 1
- `experiment_log.json` — full experiment state
