# =============================================================================
# LLM2POR Autonomous System - Configuration
# =============================================================================

import os

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Load API keys from environment variables (set in .env or system environment)
# To set: create a .env file in project root with:
#   OPENAI_API_KEY=sk-proj-...
#   GEMINI_API_KEY=AIza...
#   LLM_PROVIDER=openai
_DOTENV_LOADED = False
try:
    from dotenv import load_dotenv
    _DOTENV_LOADED = load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    pass  # python-dotenv not installed; rely on system environment variables

# *** ACTIVE LLM PROVIDER ***
# Change this to switch between providers: "openai" or "gemini"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")

# OpenAI ChatGPT API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5.2"

# Google Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3-flash-preview"  # Free tier - latest model

# Active model name (based on provider selection)
ACTIVE_MODEL = GEMINI_MODEL if LLM_PROVIDER == "gemini" else OPENAI_MODEL

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (this file's location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data files
DATA_DIR = os.path.join(BASE_DIR, "data")

MASTER_DB_PATH = os.path.join(DATA_DIR, "total_characteristics&name_singleonly_20251203.csv")
PORMAKE_5BAR_CSV_PATH = os.path.join(DATA_DIR, "total_characteristics_h2_5bar_77K.csv")

# Building Block and Topology Databases
BB_DICTIONARY_PATH = os.path.join(DATA_DIR, "pormake_bb_dictionary_v5.json")
BB_DICTIONARY_V3_PATH = BB_DICTIONARY_PATH  # Alias for backward compatibility
BB_DICTIONARY_V4_PATH = BB_DICTIONARY_PATH  # Alias for backward compatibility
BB_DICTIONARY_V5_PATH = BB_DICTIONARY_PATH  # Current version
TOPO_DICTIONARY_PATH = os.path.join(DATA_DIR, "pormake_topo_dictionary_v3.json")
TOPO_DICTIONARY_V3_PATH = TOPO_DICTIONARY_PATH  # Alias for backward compatibility

# Canonical Vocabulary (source of truth for functional group synonyms)
UNIFIED_ONTOLOGY_PATH = os.path.join(DATA_DIR, "unified_ontology.json")

# QMOF Databases for Band Gap
QMOF_CSV_PATH = os.path.join(DATA_DIR, "qmof.csv")
QMOF_TOPOLOGY_IDS_PATH = os.path.join(DATA_DIR, "qmof_ids_with_topology.txt")
QMOF_INDEX_PATH = os.path.join(DATA_DIR, "qmof_index_v2.json")
QMOF_JSONS_V3_DIR = os.path.join(DATA_DIR, "qmof_global_jsons_v3")
QMOF_BB_FILTERED_PATH = os.path.join(BASE_DIR, "..", "..", "..", "pormake_src", "dictionary_expansion", "qmofbandgap", "Processed data", "qmof-bb-filtered.json")

# hMOF Database for Gas Adsorption (H2, CH4, CO2, Xe/Kr)
HMOF_INDEX_PATH = os.path.join(DATA_DIR, "hMOF", "hmof_index.json")

# Prompt files
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
AGENT0_PROMPT_PATH = os.path.join(PROMPTS_DIR, "agent0_v3.md")  # Problem Consultant
# Agent 1 prompt: locked at v2.2.9 (2026-04-07).
# Earlier ablations (v2.3.0 with Rules A-F, v2.3.1 reflexion-only) showed no
# measurable improvement over v2.2.9 and have been retired. They remain in
# prompts/_archive/ for reproducibility of pre-v2.5 batches only.
AGENT1_PROMPT_PATH = os.path.join(PROMPTS_DIR, "agent1_v2.2.9.md")
AGENT2_PROMPT_PATH = os.path.join(PROMPTS_DIR, "agent2_v4.0.md")

# Output directory
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

# =============================================================================
# CAPABILITY MANIFEST (v3) - Describes what the system can evaluate
# It is actively imported and utilized in 
# core/agent0_handler.py
# to inform Agent 0 (the Problem Consultant) about the system's current limitations and capabilities.
# =============================================================================
CAPABILITY_MANIFEST = {
    "available_datasets": ["pormake_h2", "qmof", "hmof"],
    "available_properties": [
        "H2_uptake_77K_volumetric", "bandgap",
        "h2_uptake_2bar_77K", "h2_uptake_100bar_77K",
        "ch4_uptake_35bar_298K", "co2_uptake_2_5bar_298K",
        "xe_loading_1bar_273K", "kr_loading_1bar_273K",
        "xekr_selectivity_1bar"
    ],
    "available_geometry_fields": ["di", "df", "sa", "cv", "density", "vf", "dif",
                                   "surface_area_m2g", "void_fraction", "pld", "lcd"],
    "supported_domains": ["storage", "separation", "dac", "catalysis", "sensing",
                          "electronic", "bandgap", "gas_adsorption", "xe_kr_selectivity"],
    "execution_mode": "markscheme-driven",
    "notes": (
        "Supports H2 storage at 77K using PORMAKE database, "
        "electronic band gap prediction using QMOF database, "
        "and multi-gas adsorption (H2, CH4, CO2, Xe/Kr) using hMOF database (51K hypothetical MOFs)."
    )
}

# =============================================================================
# METRIC CONFIGURATION
# =============================================================================

# Registry of available target metrics
# Maps application-friendly names to database column names
METRIC_REGISTRY = {
    # PORMAKE H2 mode (building-block assembly)
    "h2_storage": "target",                         # Pre-mapped to 'target' in master CSV (100bar 77K)
    "h2_storage_5bar": "target",                    # 5bar 77K variant (uses PORMAKE_5BAR_CSV_PATH)
    "surface_area": "Rubre_Surface_Area",
    "void_fraction": "Void_Fraction",
    # QMOF mode (direct MOF filtering for bandgap)
    "bandgap": "outputs.pbe.bandgap",
    # hMOF mode (direct MOF filtering for gas adsorption)
    "h2_uptake_100bar": "h2_uptake_100bar_77K",
    "h2_uptake_2bar": "h2_uptake_2bar_77K",
    "ch4_storage": "ch4_uptake_35bar_298K",
    "co2_capture": "co2_uptake_2_5bar_298K",
    "xe_kr_selectivity": "xekr_selectivity_1bar",
    "xe_loading": "xe_loading_1bar_273K",
    "kr_loading": "kr_loading_1bar_273K",
}

# Metrics that belong to hMOF mode
_HMOF_METRICS = {
    "h2_uptake_100bar_77K", "h2_uptake_2bar_77K",
    "ch4_uptake_35bar_298K", "co2_uptake_2_5bar_298K",
    "xekr_selectivity_1bar", "xe_loading_1bar_273K", "kr_loading_1bar_273K",
}

# The currently active metric for optimization
# Default: H2 Storage (column renamed to 'target' in master DB)
ACTIVE_METRIC_COLUMN = "target"


def is_qmof_mode() -> bool:
    """Check if the system is running in QMOF (band gap) mode."""
    return ACTIVE_METRIC_COLUMN == METRIC_REGISTRY.get("bandgap", "outputs.pbe.bandgap")


def is_hmof_mode() -> bool:
    """Check if the system is running in hMOF (gas adsorption) mode."""
    return ACTIVE_METRIC_COLUMN in _HMOF_METRICS


# Track which PorMake markscheme variant is active (set by run_experiment.py)
_PORMAKE_5BAR_ACTIVE = False


def is_pormake_5bar_mode() -> bool:
    """Check if the system is using the 5bar 77K H2 markscheme."""
    return _PORMAKE_5BAR_ACTIVE


def get_master_db_path() -> str:
    """Return the active PorMake markscheme CSV path."""
    if _PORMAKE_5BAR_ACTIVE:
        return PORMAKE_5BAR_CSV_PATH
    return MASTER_DB_PATH


def get_agent1_prompt_path() -> str:
    """Return the Agent 1 prompt path.

    v2.5 (2026-04-07): Locked at v2.2.9 for all three database modes
    (PORMAKE / QMOF / hMOF). The earlier v2.3.0 and v2.3.1 ablations
    produced no measurable improvement and are retired to
    prompts/_archive/ for batch reproducibility only.
    """
    return AGENT1_PROMPT_PATH


# hMOF column mapping: hmof_index property names → sensitivity analyzer column names
HMOF_COLUMN_MAP = {
    "lcd": "di",              # largest cavity diameter
    "pld": "df",              # pore limiting diameter
    "surface_area_m2g": "sa", # surface area (m2/g)
    "void_fraction": "vf",    # void fraction
    "density": "density",     # crystal density (g/cm3)
}


def validate_api_keys():
    """Validate that the required API key is present for the active provider."""
    dotenv_hint = ""
    if not _DOTENV_LOADED:
        dotenv_hint = " (python-dotenv may not be installed — run: pip install python-dotenv)"

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise ValueError(
            f"OPENAI_API_KEY not set.{dotenv_hint} "
            "Add it to .env file or set as environment variable."
        )
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise ValueError(
            f"GEMINI_API_KEY not set.{dotenv_hint} "
            "Add it to .env file or set as environment variable."
        )


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# Default design inquiry
DEFAULT_INQUIRY = "Design a MOF for high capacity Hydrogen storage at 77K"

# Feedback sample sizes (as per pilot notebook)
FEEDBACK_SAMPLE_SIZE = 8  # Budget-matched: 8 samples x 4 beams x 10 iters = 320 ~ BO@300
FEEDBACK_SAMPLE_SIZE_LARGE = 30  # For Blind Random, Best vs Worst

# Sampling mode: stochastic (different samples each iteration)
STOCHASTIC_SAMPLING = True

# Agent 0 interview settings
AGENT0_MAX_TURNS = 10
AGENT0_SKIP_COMMANDS = ['proceed', 'skip', 'done', 'go']

# =============================================================================
# LLM CLIENT SETTINGS
# =============================================================================

# Maximum output tokens for LLM responses
# Agent 1 outputs ~7 rich-text JSON fields; needs enough room for verbose models
LLM_MAX_OUTPUT_TOKENS = 32000

# Retry and timeout settings for Gemini REST API
LLM_MAX_RETRIES = 2
LLM_REQUEST_TIMEOUT = 120  # seconds

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

# Sensitivity report columns (matching pilot notebook format)
# Definition of Band Gap Categories (SP-3.01 Hybrid Fix)
# Boundaries: lower bound INCLUSIVE (>=), upper bound EXCLUSIVE (<)
# Continuous statistics (Q1, Q3, Min, Max) reported alongside categories
BANDGAP_CATEGORIES = {
    "Metallic (<0.1eV)": (0.0, 0.1),
    "Narrow/IR (0.1-1.6eV)": (0.1, 1.6),
    "Vis Red/Yel (1.6-2.2eV)": (1.6, 2.2),
    "Vis Blue/Vio (2.2-3.1eV)": (2.2, 3.1),
    "UV Active (3.1-4.0eV)": (3.1, 4.0),
    "Insulator (>=4.0eV)": (4.0, float('inf'))
}

# Dynamic Sensitivity Columns
def get_sensitivity_columns() -> list:
    """Returns report columns based on the active metric."""
    m = ACTIVE_METRIC_COLUMN
    
    # QMOF Contextual Override
    if m == METRIC_REGISTRY.get("bandgap", "outputs.pbe.bandgap"):
        return [
            "Filter", "Count", "% Removed",
            "Metallic (<0.1eV)", "Narrow/IR (0.1-1.6eV)", 
            "Vis Red/Yel (1.6-2.2eV)", "Vis Blue/Vio (2.2-3.1eV)", 
            "UV Active (3.1-4.0eV)", "Insulator (>=4.0eV)",
            "Median Bandgap", "Q1 Bandgap", "Q3 Bandgap",
            "Min Bandgap", "Max Bandgap",
            "P-Value"
        ]
        
    # Standard H2 Mode
    label = "Performance" 
    for key, val in METRIC_REGISTRY.items():
        if val == m:
            label = key.replace("_", " ").title()
            break
            
    return [
        "Filter", "Count", "% Removed", 
        f"Avg Top 5 ({label})", f"Avg Worse 5 ({label})", 
        f"Best {label}", f"Median {label}", 
        "EF @ 1%", "EF @ 5%", "P-Value"
    ]

