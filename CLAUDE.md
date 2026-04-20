# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM2POR** is an autonomous Metal-Organic Framework (MOF) designer. LLMs (GPT/Gemini) iteratively propose MOF designs evaluated against real computational databases: QMOF (20K MOFs, electronic properties), hMOF (51K hypothetical MOFs, gas adsorption), and PORMAKE (component assembly from building blocks).

## Commands

```bash
# Setup
python -m venv llm2auto && source llm2auto/bin/activate
pip install -r requirements.txt
pip install -e .                 # Required before running simulation pipeline
git lfs pull                    # Download large database files (~175 MB)

# Run experiment
python run_experiment.py

# Run a single module test (no pytest — each module has embedded test_<module>())
python core/llm_client.py              # test_llm_client()
python core/agent0_handler.py          # test_agent0()
python core/agent1_handler.py          # test_agent1()
python core/agent2_handler.py          # test_agent2()
python core/agent3_260324.py           # test_agent3()
python core/matchmaker.py              # test_matchmaker()
python core/feedback_generator.py      # test_feedback_generator()
python core/sensitivity_analyzer.py   # test_sensitivity_analyzer()
python core/memory_manager.py          # test_memory_and_logger()
```

### Environment setup (`.env`)
```
LLM_PROVIDER=openai           # or "gemini"
OPENAI_API_KEY=sk-proj-...
# GEMINI_API_KEY=...          # if using Gemini
```

### Optional Simulation Pipeline

```bash
# Full pipeline: matchmaker → filter → generate → optimize → RASPA3
python core/run_simulation.py --input_json experiments/exp_XXX/agent2_output.json \
    --output_dir experiments/exp_XXX/simulation --num_mofs 10

# Individual steps
python core/filter_candidate.py --input_json /path/to/agent2_output.json --output_dir /path/to/output
python core/simulation/generate_mofs.py --input_json /path/to/filtered.json --output_dir /path/to/cifs
python core/simulation/opt/optimize.py --cif-dir /path/to/cifs   # full LAMMPS pipeline
python core/simulation/opt/optimize.py --cif-dir /path/to/cifs --interface-only
python core/simulation/opt/optimize.py --cif-dir /path/to/cifs --optimize-only
python core/simulation/opt/optimize.py --cif-dir /path/to/cifs --convert-only
python core/simulation/gcmc/run_raspa.py --mof-dir /path/to/cifs --output-dir /path/to/results
python core/simulation/gcmc/analyze.py --output-dir /path/to/results
```

## Architecture

### The Autonomous Loop

Each experiment iteration runs these steps in sequence:

```
User Inquiry
    ↓
[Agent 0] Problem Consultant (optional interview, agent0_handler.py)
    ↓
[Agent 1] Hypothesis Generator — multi-turn, stateful (agent1_handler.py)
    ↓  JSON hypothesis
[Agent 2] Constraint Extractor — stateless translation to DB search format (agent2_handler.py)
    ↓  quantitative constraints
[Matchmaker] Component/MOF Discovery — auto-selects database based on inquiry mode
    ↓  ranked candidates
[Sensitivity Analyzer] Enrichment Factor + p-value evaluation (sensitivity_analyzer.py)
    ↓  statistical metrics
[Feedback Generator] 4-beam feedback → Agent 1 refines hypothesis (feedback_generator.py)
```

### Database Modes (auto-detected from inquiry keywords)

| Mode | Matchmaker | Database | Target Property |
|------|------------|----------|-----------------|
| PORMAKE | `matchmaker.py` | 50K+ building blocks | H2 storage, custom |
| QMOF | `qmof_matchmaker.py` | 20,373 MOFs | Band gap |
| hMOF | `hmof_matchmaker.py` | 51,163 MOFs | CH4, CO2, Xe/Kr, H2 adsorption |

Mode-detection functions: `is_qmof_mode()`, `is_hmof_mode()` in `config.py`.

### Constraint Engine: Branched Hypothesis Matching

Agent 2 outputs a `linker_branches` schema supporting **AND-within-branch** + **OR-between-branches** logic, preserving alternative linker strategies that Agent 1 proposes:

```json
{
  "linker_query": {
    "linker_branches": [
      {"description": "pyridine dicarboxylate", "required_tags": ["Pyridine", "Carboxyl"]},
      {"description": "azolate",                "required_tags": ["Azolate"]}
    ]
  }
}
```

PORMAKE strips coordination tags (`Carboxyl`, `Carbonyl`, `Phosphonate`, `Sulfonate`) from branches — in PORMAKE's grammar these live on the Node SBU, not the Edge linker. QMOF and hMOF do not strip.

### 4-Beam Diagnostic Feedback

The default feedback runs four parallel searches per iteration:

| Database | Beam 1 | Beam 2 | Beam 3 | Beam 4 | Purpose |
|----------|--------|--------|--------|--------|---------|
| PORMAKE/hMOF | Full hypothesis (Z) | Chemistry only (A) | Metal only (F) | Global baseline | Is geometry or chemistry the bottleneck? |
| QMOF | Full hypothesis (Z) | Metal control (F) | Linker control (G) | Global baseline | Metal d-electrons vs. linker conjugation? |

Agent 1 is **blinded**: anonymous `MOF-1, MOF-2, ...` labels, no database names, generic beam headers — prevents inferring which database is active.

### Key Design Decisions

- **Agent 1 is stateful**: Maintains multi-turn conversation history. All prior hypotheses and feedback are visible in the `LLMClient.messages` buffer. Must list all constraints explicitly every iteration (no external memory injection — the Scientific Journal was removed in v2.5).
- **Agent 2 is stateless**: Single-shot translation; validated against canonical vocabulary from `data/unified_ontology.json` (80+ approved functional group tags).
- **Feedback is hidden from agents**: Enrichment Factor and p-values are for human analysis only; Agent 1 receives only structured qualitative/quantitative feedback.
- **No pytest**: Tests are `if __name__ == "__main__"` blocks executed by running the module directly.
- **All config in `config.py`**: Database paths, API settings, metric registry, display settings — all centralized here.
- **Use `os.path` not pathlib** throughout the codebase.
- **Agent 1 prompt locked at `agent1_v2.2.9.md`** (v2.5): Three-way ablation (v2.2.9 / v2.3.0 / v2.3.1) showed no measurable improvement from the v2.3.x variants.

### Critical Files

| File | Role |
|------|------|
| `run_experiment.py` | Main orchestrator — experiment loop, user interaction |
| `config.py` | All configuration, paths, capability manifest, metric registry |
| `core/llm_client.py` | Unified OpenAI/Gemini wrapper; use `LLMClient.extract_json()` for robust JSON parsing |
| `core/constraint_utils.py` | Tag/ontology parsing + AND/OR/NOT/branch logic shared by matchmakers and SA |
| `prompts/agent1_v2.2.9.md` | Agent 1 system prompt — locked at v2.5 |
| `prompts/agent2_v4.0.md` | Agent 2 system prompt — enforces canonical vocabulary + `linker_branches` schema |
| `data/unified_ontology.json` | Source of truth for functional group vocabulary |

### Key Configuration Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MAX_OUTPUT_TOKENS` | 32000 | Max tokens for LLM response |
| `LLM_REQUEST_TIMEOUT` | 120 | API timeout in seconds |
| `FEEDBACK_SAMPLE_SIZE` | 8 | Samples per beam (8 × 4 beams × 10 iters = 320 total) |
| `STOCHASTIC_SAMPLING` | True | Different samples each iteration |
| `AGENT1_PROMPT_PATH` | `prompts/agent1_v2.2.9.md` | Active Agent 1 prompt |
| `AGENT2_PROMPT_PATH` | `prompts/agent2_v4.0.md` | Active Agent 2 prompt |

### Output Structure

Experiments write to `experiments/exp_YYYYMMDD_HHMM_{mode}/`:
- `raw_user_input.txt` — original inquiry
- `experiment_log.txt` — full run log
- `iteration_N/` — per-iteration outputs (hypothesis, constraints, sensitivity reports)

## Code Style

- **Line length**: 120 characters max
- **Imports**: stdlib → third-party → local, blank lines between groups; use `sys.path.insert(0, ...)` for local imports
- **Naming**: `snake_case` modules/functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- **Type hints**: required on all function signatures
- **Docstrings**: Google-style (Args / Returns / Raises)
- **Console output**: bracket notation — `print("[Agent 1] Generating hypothesis...")`
- **JSON output**: `json.dump(data, f, indent=2, ensure_ascii=False)` for logs; `LLMClient.extract_json()` for LLM response parsing
- **Error handling**: use specific exception types, fail fast — never `except: pass`
- **Section headers**: `# ====...====` pattern in module files
