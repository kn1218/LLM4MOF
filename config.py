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

    _DOTENV_LOADED = load_dotenv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    )
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

MASTER_DB_PATH = os.path.join(DATA_DIR, "total_characteristics_h2_100bar_77K.csv")

# PorMake H2 unit variants — 2 pressures × 3 units = 6 CSVs
# 100bar variants (MASTER_DB_PATH is the volumetric default)
PORMAKE_100BAR_GPERL_CSV_PATH = os.path.join(DATA_DIR, "total_characteristics_h2_100bar_77K_gperL.csv")
PORMAKE_100BAR_MOLKG_CSV_PATH = os.path.join(DATA_DIR, "total_characteristics_h2_100bar_77K_mol_kg.csv")
# 5bar variants
PORMAKE_5BAR_CSV_PATH = os.path.join(DATA_DIR, "total_characteristics_h2_5bar_77K.csv")
PORMAKE_5BAR_GPERL_CSV_PATH = os.path.join(DATA_DIR, "total_characteristics_h2_5bar_77K_gperL.csv")
PORMAKE_5BAR_MOLKG_CSV_PATH = os.path.join(DATA_DIR, "total_characteristics_h2_5bar_77K_mol_kg.csv")

# Building Block and Topology Databases
BB_DICTIONARY_PATH = os.path.join(DATA_DIR, "pormake_bb_dictionary_v6.json")
BB_DICTIONARY_V3_PATH = BB_DICTIONARY_PATH  # Alias for backward compatibility
BB_DICTIONARY_V4_PATH = BB_DICTIONARY_PATH  # Alias for backward compatibility
BB_DICTIONARY_V5_PATH = BB_DICTIONARY_PATH  # Alias for backward compatibility
BB_DICTIONARY_V6_PATH = BB_DICTIONARY_PATH  # Current version
TOPO_DICTIONARY_PATH = os.path.join(DATA_DIR, "pormake_topo_dictionary_v3.json")
TOPO_DICTIONARY_V3_PATH = TOPO_DICTIONARY_PATH

# Canonical Vocabulary (source of truth for functional group synonyms)
UNIFIED_ONTOLOGY_PATH = os.path.join(DATA_DIR, "unified_ontology.json")

# QMOF Databases for Band Gap
QMOF_CSV_PATH = os.path.join(DATA_DIR, "qmof.csv")
QMOF_TOPOLOGY_IDS_PATH = os.path.join(DATA_DIR, "qmof_ids_with_topology.txt")
QMOF_INDEX_PATH = os.path.join(DATA_DIR, "qmof_index_v2.json")
QMOF_JSONS_V3_DIR = os.path.join(DATA_DIR, "qmof_global_jsons_v3")
QMOF_BB_FILTERED_PATH = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "..",
    "pormake_src",
    "dictionary_expansion",
    "qmofbandgap",
    "Processed data",
    "qmof-bb-filtered.json",
)

# hMOF Database for Gas Adsorption (H2, CH4, CO2, Xe/Kr)
HMOF_INDEX_PATH = os.path.join(DATA_DIR, "hMOF", "hmof_index.json")

# Prompt files
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
AGENT0_PROMPT_PATH = os.path.join(PROMPTS_DIR, "agent0_v3.md")  # Problem Consultant
# Agent 1 prompt: v3.0 (2026-04-14).
# v3.0 is fully database/application-agnostic. All DB-specific context
# (metric, unit, pressure, structural constraints) is injected at runtime
# by agent1_handler._build_system_context(). Prior versions (v2.2.9, v2.3.x)
# are in prompts/_archive/ for reproducibility of pre-v3.0 batches.
# v3.1 (2026-04-15): Restores concrete examples, incremental constraint discipline,
# specific beam descriptions from v2.2.9. Keeps v3.0's unit awareness and
# mechanism chain. Removes database-specific constraints (ditopic, scarce features)
# from prompt — those belong in the user query if needed.
AGENT1_PROMPT_PATH = os.path.join(PROMPTS_DIR, "agent1_v2.2.9.2.md")
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
        # PorMake H2 — 2 pressures × 3 units
        "H2_uptake_100bar_77K_volumetric",   # cm³(STP)/cm³
        "H2_uptake_100bar_77K_gperL",        # g/L
        "H2_uptake_100bar_77K_gravimetric",  # mol/kg
        "H2_uptake_5bar_77K_volumetric",     # cm³(STP)/cm³
        "H2_uptake_5bar_77K_gperL",          # g/L
        "H2_uptake_5bar_77K_gravimetric",    # mol/kg
        # QMOF
        "bandgap",
        # hMOF
        "h2_uptake_2bar_77K", "h2_uptake_100bar_77K",
        "ch4_uptake_35bar_298K", "co2_uptake_2_5bar_298K",
        "xe_loading_1bar_273K", "kr_loading_1bar_273K",
        "xekr_selectivity_1bar"
    ],
    "execution_mode": "markscheme-driven",
    "notes": (
        "Supports H2 storage at 77K using PORMAKE database (100bar and 5bar conditions) "
        "in three unit variants: cm³(STP)/cm³ (volumetric), g/L, and mol/kg (gravimetric). "
        "Electronic band gap prediction using QMOF database. "
        "Multi-gas adsorption (H2, CH4, CO2, Xe/Kr) using hMOF database (51K hypothetical MOFs)."
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
    "h2_uptake_100bar_77K",
    "h2_uptake_2bar_77K",
    "ch4_uptake_35bar_298K",
    "co2_uptake_2_5bar_298K",
    "xekr_selectivity_1bar",
    "xe_loading_1bar_273K",
    "kr_loading_1bar_273K",
}

# The currently active metric for optimization
# Default: H2 Storage (column renamed to 'target' in master DB)
ACTIVE_METRIC_COLUMN = "target"

# =============================================================================
# UNIT REGISTRY — maps metric columns to their display unit and conversion info
# =============================================================================
# Each entry: column_name -> {display, type}
#   display: human-readable unit string for feedback tables
#   type: unit category (volumetric, gravimetric, energy, ratio, etc.)
UNIT_REGISTRY: dict[str, dict[str, str]] = {
    # PorMake H2 (100bar & 5bar) — pre-computed in cm³(STP)/cm³ (volumetric)
    "target": {"display": "cm³(STP)/cm³", "type": "volumetric"},
    # QMOF band gap — eV
    "outputs.pbe.bandgap": {"display": "eV", "type": "energy"},
    # hMOF gas uptakes — cm³(STP)/g (gravimetric per gram, Wilmer et al. 2012)
    # NOTE: hMOF is gravimetric (per gram), PorMake is volumetric (per cm³ framework)
    "h2_uptake_100bar_77K": {"display": "cm³(STP)/g", "type": "gravimetric"},
    "h2_uptake_2bar_77K": {"display": "cm³(STP)/g", "type": "gravimetric"},
    "ch4_uptake_35bar_298K": {"display": "cm³(STP)/g", "type": "gravimetric"},
    "co2_uptake_2_5bar_298K": {"display": "cm³(STP)/g", "type": "gravimetric"},
    # hMOF Xe/Kr loading — mol/kg (molar gravimetric)
    "xe_loading_1bar_273K": {"display": "mol/kg", "type": "gravimetric_molar"},
    "kr_loading_1bar_273K": {"display": "mol/kg", "type": "gravimetric_molar"},
    # hMOF Xe/Kr selectivity — dimensionless ratio
    "xekr_selectivity_1bar": {"display": "dimensionless", "type": "ratio"},
}

# Molar volume at STP: 22414 cm³/mol = 22.414 L/mol
MOLAR_VOL_STP_CM3_PER_MMOL = 22.414  # cm³(STP) per mmol → used as: mol/kg * g/cm³ * 22.414 = cm³(STP)/cm³


def get_active_unit() -> str:
    """Return the display unit string for the currently active metric.

    When a PorMake unit variant is active, overrides the default
    volumetric unit for 'target' column.
    """
    if ACTIVE_METRIC_COLUMN == "target":
        if _PORMAKE_GRAVIMETRIC_ACTIVE:
            return "mol/kg"
        if _PORMAKE_GPERL_ACTIVE:
            return "g/L"
    entry = UNIT_REGISTRY.get(ACTIVE_METRIC_COLUMN)
    if entry:
        return entry["display"]
    return ""


def get_active_unit_type() -> str:
    """Return the unit type (volumetric, gravimetric, energy, ratio) for the active metric."""
    if ACTIVE_METRIC_COLUMN == "target":
        if _PORMAKE_GRAVIMETRIC_ACTIVE:
            return "gravimetric"
        if _PORMAKE_GPERL_ACTIVE:
            return "volumetric_mass"
    entry = UNIT_REGISTRY.get(ACTIVE_METRIC_COLUMN)
    if entry:
        return entry["type"]
    return "unknown"


def is_qmof_mode() -> bool:
    """Check if the system is running in QMOF (band gap) mode."""
    return ACTIVE_METRIC_COLUMN == METRIC_REGISTRY.get("bandgap", "outputs.pbe.bandgap")


def is_hmof_mode() -> bool:
    """Check if the system is running in hMOF (gas adsorption) mode."""
    return ACTIVE_METRIC_COLUMN in _HMOF_METRICS


# Track which PorMake markscheme variant is active (set by run_experiment.py)
# Pressure selector
_PORMAKE_5BAR_ACTIVE = False
# Unit selectors (at most one should be True; both False → volumetric cm³(STP)/cm³)
_PORMAKE_GRAVIMETRIC_ACTIVE = False  # mol/kg
_PORMAKE_GPERL_ACTIVE = False        # g/L


def is_pormake_5bar_mode() -> bool:
    """Check if the system is using the 5bar 77K H2 markscheme."""
    return _PORMAKE_5BAR_ACTIVE


def is_pormake_gravimetric_mode() -> bool:
    """Check if the system is using the gravimetric (mol/kg) PorMake markscheme."""
    return _PORMAKE_GRAVIMETRIC_ACTIVE


def is_pormake_gperL_mode() -> bool:
    """Check if the system is using the g/L PorMake markscheme."""
    return _PORMAKE_GPERL_ACTIVE


def get_master_db_path() -> str:
    """Return the active PorMake markscheme CSV path.

    Routes based on pressure (5bar vs 100bar) × unit (volumetric, g/L, mol/kg).
    """
    if _PORMAKE_5BAR_ACTIVE:
        if _PORMAKE_GPERL_ACTIVE:
            return PORMAKE_5BAR_GPERL_CSV_PATH
        if _PORMAKE_GRAVIMETRIC_ACTIVE:
            return PORMAKE_5BAR_MOLKG_CSV_PATH
        return PORMAKE_5BAR_CSV_PATH
    # 100bar (default pressure)
    if _PORMAKE_GPERL_ACTIVE:
        return PORMAKE_100BAR_GPERL_CSV_PATH
    if _PORMAKE_GRAVIMETRIC_ACTIVE:
        return PORMAKE_100BAR_MOLKG_CSV_PATH
    return MASTER_DB_PATH


def get_agent1_prompt_path() -> str:
    """Return the Agent 1 prompt path.

    v2.2.9.2 (2026-04-15): Production prompt. Database/application-agnostic
    with concrete examples and incremental constraint discipline.
    Prior versions in prompts/_archive/ for reproducibility.
    """
    return AGENT1_PROMPT_PATH


# hMOF column mapping: hmof_index property names → sensitivity analyzer column names
HMOF_COLUMN_MAP = {
    "lcd": "di",  # largest cavity diameter
    "pld": "df",  # pore limiting diameter
    "surface_area_m2g": "sa",  # surface area (m2/g)
    "void_fraction": "vf",  # void fraction
    "density": "density",  # crystal density (g/cm3)
}


def validate_api_keys():
    """Validate that the required API key is present for the active provider."""
    dotenv_hint = ""
    if not _DOTENV_LOADED:
        dotenv_hint = (
            " (python-dotenv may not be installed — run: pip install python-dotenv)"
        )

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
AGENT0_SKIP_COMMANDS = ["proceed", "skip", "done", "go"]

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
# MOF2ZEO CONFIGURATION (Agent 3 - Geometry Prediction)
# =============================================================================
# mof2zeo is a PyTorch Lightning model that predicts MOF geometric descriptors
# (Di, Df, SA, VF, density, CV, Dif) from (topology, node, edge) triples.
# Used by core/filter_candidate.py to rank matchmaker candidates before CIF
# build + LAMMPS + RASPA3 simulation in core/run_simulation.py.
# Model trained to predict: Di, Df, SA, VF, density, CV, Dif from MOF components.


MOF2ZEO_DIR = os.path.join(BASE_DIR, "core", "mof2zeo")

# Config file for model hyperparameters
MOF2ZEO_CONFIG_PATH = os.path.join(MOF2ZEO_DIR, "config.yaml")

# Trained checkpoint (76 MB, Git LFS)
MOF2ZEO_CKPT_PATH = os.path.join(MOF2ZEO_DIR, "ckpt", "epoch=478-step=213634.ckpt")

# Scaler files for inverse transform (mean/std from training data)
MOF2ZEO_SCALER_MEAN_PATH = os.path.join(MOF2ZEO_DIR, "scaler", "mean_all.csv")
MOF2ZEO_SCALER_STD_PATH = os.path.join(MOF2ZEO_DIR, "scaler", "std_all.csv")

# Vocabulary files (topology, node, edge class mappings)
MOF2ZEO_TOPOLOGY_FILE = os.path.join(MOF2ZEO_DIR, "data", "topology.txt")
MOF2ZEO_NODE_FILE = os.path.join(MOF2ZEO_DIR, "data", "node.txt")
MOF2ZEO_EDGE_FILE = os.path.join(MOF2ZEO_DIR, "data", "edge.txt")
MOF2ZEO_FEATURE_FILE = os.path.join(MOF2ZEO_DIR, "data", "feature_name.txt")

# Model hyperparameters (mirror core/mof2zeo/config.yaml)
MOF2ZEO_LATENT_DIM = 128
MOF2ZEO_HID_DIM1 = 64
MOF2ZEO_HID_DIM2 = 32
MOF2ZEO_DESC_DIM = 7  # sa, cv, density, vf, di, df, dif


def is_mof2zeo_available() -> bool:
    """Check whether the mof2zeo checkpoint and scaler files exist on disk."""
    return os.path.exists(MOF2ZEO_CKPT_PATH) and os.path.exists(MOF2ZEO_SCALER_MEAN_PATH)


# =============================================================================
# LIVE SIMULATION CONFIGURATION (Han pipeline as feedback source)
# =============================================================================
# These settings control the live-simulation loop (run_live_experiment.py).
# The markscheme path (run_experiment.py) is unaffected.

LIVE_SIM_N_PER_BEAM = 8            # Target successful simulations per beam
LIVE_SIM_N_BEAMS = 4               # Z (full), A (chem-only), F (metal-only), Total (random)
LIVE_SIM_POOL_MULTIPLIER = 3       # Pool size = N_PER_BEAM * POOL_MULTIPLIER (refill budget)
LIVE_SIM_POOL_MULTIPLIER_RANDOM = 4  # Larger pool for Beam 4 (random baseline, higher failure rate)
LIVE_SIM_MIN_SUCCESSES = 4         # Accept partial beam if >= this many successes

LIVE_SIM_RASPA_CYCLES = 15000      # Production: 15k cycles (Han's production setting)
LIVE_SIM_RASPA_INIT_CYCLES = 5000  # Production: 5k init cycles (Han's default)
LIVE_SIM_RASPA_TEMPERATURE = 77.0  # K (hydrogen storage standard)
LIVE_SIM_RASPA_PRESSURE = 10000000.0  # Pa (~100 bar)

LIVE_SIM_SKIP_LAMMPS = False       # LAMMPS enabled on HPC (dirac1); local smoke tests override
LIVE_SIM_LAMMPS_TIMEOUT = 900      # 15 min cap per MOF
LIVE_SIM_RASPA_TIMEOUT = 7200      # 2 hours — empirical test showed 94 min max for 15k cycles

LIVE_SIM_MOF2ZEO_TOPN = 50         # mof2zeo top-N candidates per beam before sampling
LIVE_SIM_CACHE_DIR = os.path.join(BASE_DIR, "experiments")

LIVE_SIM_MAX_ITERATIONS = 10       # 10 iterations for production run

# HPC Configuration (dirac1 cluster)
HPC_HOST = "dirac1"                 # SSH hostname (must be in ~/.ssh/config)
HPC_BASE_DIR = "~/llm2por"          # Base directory on HPC
HPC_NODE_PROPERTY = "ac"            # PBS node property for qsub
HPC_POLL_INTERVAL = 300             # 5 minutes between SSH polls
HPC_POLL_MAX_HOURS = 5              # Give up polling after this many hours
HPC_SSH_RETRIES = 3                 # Retry SSH on connection failure
HPC_SSH_RETRY_DELAYS = [30, 60, 120]  # Exponential backoff (seconds)


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
    "Insulator (>=4.0eV)": (4.0, float("inf")),
}


# Dynamic Sensitivity Columns
def get_sensitivity_columns() -> list:
    """Returns report columns based on the active metric."""
    m = ACTIVE_METRIC_COLUMN

    # QMOF Contextual Override
    if m == METRIC_REGISTRY.get("bandgap", "outputs.pbe.bandgap"):
        return [
            "Filter",
            "Count",
            "% Removed",
            "Metallic (<0.1eV)",
            "Narrow/IR (0.1-1.6eV)",
            "Vis Red/Yel (1.6-2.2eV)",
            "Vis Blue/Vio (2.2-3.1eV)",
            "UV Active (3.1-4.0eV)",
            "Insulator (>=4.0eV)",
            "Median Bandgap",
            "Q1 Bandgap",
            "Q3 Bandgap",
            "Min Bandgap",
            "Max Bandgap",
            "P-Value",
        ]

    # Standard H2 Mode
    label = "Performance"
    for key, val in METRIC_REGISTRY.items():
        if val == m:
            label = key.replace("_", " ").title()
            break

    return [
        "Filter",
        "Count",
        "% Removed",
        f"Avg Top 5 ({label})",
        f"Avg Worse 5 ({label})",
        f"Best {label}",
        f"Median {label}",
        "EF @ 1%",
        "EF @ 5%",
        "P-Value",
    ]
