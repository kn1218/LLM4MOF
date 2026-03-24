"""
Strategy Registry for LLM vs Optimization Comparison Study.

Each strategy produces a constraints dict (same schema as Agent 2 output)
that feeds into the same matchmaker + sensitivity analyzer pipeline.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

STRATEGIES = {
    "v229": {
        "prompt": os.path.join(PROMPTS_DIR, "agent1_v2.2.9.md"),
        "mode": "llm",
        "routing": "fixed",  # same prompt for all DB modes
        "label": "Baseline LLM (v2.2.9)",
        "description": "No reasoning rules, no structured reflection",
    },
    "v230": {
        "prompt": None,  # resolved dynamically via per-DB routing
        "mode": "llm",
        "routing": "per_db",  # uses pormake/qmof/hmof variants
        "label": "Reasoning Rules (v2.3.0)",
        "description": "Rules A-F + per-DB rules G/H",
        "prompt_map": {
            "pormake": os.path.join(PROMPTS_DIR, "agent1_v2.3.0.md"),
            "qmof": os.path.join(PROMPTS_DIR, "agent1_v2.3.0_qmof.md"),
            "hmof": os.path.join(PROMPTS_DIR, "agent1_v2.3.0_hmof.md"),
        },
    },
    "v231": {
        "prompt": os.path.join(PROMPTS_DIR, "agent1_v2.3.1_reflexion_only.md"),
        "mode": "llm",
        "routing": "fixed",  # universal prompt
        "label": "Reflexion Only (v2.3.1)",
        "description": "Structured reflection, no reasoning rules, universal",
    },
    "random": {
        "prompt": None,
        "mode": "baseline",
        "routing": None,
        "label": "Random Search",
        "description": "Random sampling with no memory between iterations",
    },
    "lhs": {
        "prompt": None,
        "mode": "baseline",
        "routing": None,
        "label": "Latin Hypercube Sampling",
        "description": "Systematic space-filling design, no learning",
    },
    "bo": {
        "prompt": None,
        "mode": "baseline",
        "routing": None,
        "label": "Bayesian Optimization (TPE)",
        "description": "Optuna TPE surrogate model with expected improvement",
    },
    "ga": {
        "prompt": None,
        "mode": "baseline",
        "routing": None,
        "label": "Genetic Algorithm",
        "description": "Population-based evolutionary search with crossover/mutation",
    },
}


def get_strategy(name: str) -> dict:
    """Get strategy config by name. Raises ValueError if not found."""
    if name not in STRATEGIES:
        valid = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{name}'. Valid: {valid}")
    return {**STRATEGIES[name], "name": name}


def get_prompt_for_strategy(strategy_name: str, db_mode: str = "pormake") -> str:
    """Resolve the prompt path for a strategy and database mode.

    Args:
        strategy_name: One of the STRATEGIES keys
        db_mode: "pormake", "qmof", or "hmof"

    Returns:
        Absolute path to the prompt file, or None for baseline strategies
    """
    strategy = get_strategy(strategy_name)
    if strategy["mode"] == "baseline":
        return None
    if strategy["routing"] == "per_db":
        return strategy["prompt_map"].get(db_mode, strategy["prompt_map"]["pormake"])
    return strategy["prompt"]


def get_all_strategy_names() -> list:
    """Return all strategy names."""
    return list(STRATEGIES.keys())


def get_llm_strategies() -> list:
    """Return strategy names that use LLM (require API calls)."""
    return [k for k, v in STRATEGIES.items() if v["mode"] == "llm"]


def get_baseline_strategies() -> list:
    """Return strategy names that don't use LLM (no API calls)."""
    return [k for k, v in STRATEGIES.items() if v["mode"] == "baseline"]
