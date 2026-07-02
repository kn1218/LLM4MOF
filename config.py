# =============================================================================
# LLM4MOF Autonomous System - Configuration
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
# Canonical building-block whitelist = the retrained mof2zeo vocabulary in core/mof2zeo/data/
# (SINGLE source of truth). The matchmaker proposes only from these and the markscheme DBs in
# data/ are pre-filtered to them (scripts/build_canonical_db.py), so the agent never proposes a
# building block mof2zeo cannot predict. (Previously duplicated under data/bblist/ — removed to
# avoid drift; the only diff was N621, absent from the retrained mof2zeo vocab.)
BBLIST_DIR = os.path.join(BASE_DIR, "core", "mof2zeo", "data")
BBLIST_NODE_PATH = os.path.join(BBLIST_DIR, "node.txt")
BBLIST_EDGE_PATH = os.path.join(BBLIST_DIR, "edge.txt")
BBLIST_TOPOLOGY_PATH = os.path.join(BBLIST_DIR, "topology.txt")

BB_DICTIONARY_PATH = os.path.join(DATA_DIR, "pormake_bb_dictionary_v7.json")
BB_DICTIONARY_V3_PATH = BB_DICTIONARY_PATH  # Alias (used by matchmaker, name_resolver, run_experiment)
BB_DICTIONARY_V7_PATH = BB_DICTIONARY_PATH  # Current version (adds llm_additions inside functional_groups_categorized)
TOPO_DICTIONARY_PATH = os.path.join(DATA_DIR, "pormake_topo_dictionary_v3.json")
TOPO_DICTIONARY_V3_PATH = TOPO_DICTIONARY_PATH

# Canonical Vocabulary (source of truth for functional group synonyms)
UNIFIED_ONTOLOGY_PATH = os.path.join(DATA_DIR, "unified_ontology.json")

# QMOF Databases for Band Gap
QMOF_CSV_PATH = os.path.join(DATA_DIR, "qmof.csv")
QMOF_INDEX_PATH = os.path.join(DATA_DIR, "qmof_index_v2.json")

# hMOF Database for Gas Adsorption (H2, CH4, CO2, Xe/Kr)
HMOF_INDEX_PATH = os.path.join(DATA_DIR, "hMOF", "hmof_index.json")

# Prompt files
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
# Agent 1: v3.0_production — axis-neutral / direction-symmetric universal prompt with a
#   SOFT decoration commit (require presence or <=2, never a high hard min_group_counts).
AGENT1_PROMPT_PATH = os.path.join(PROMPTS_DIR, "agent1_v3.0_production.md")
# Agent 2: v4.1 — adds the PORMAKE single-building-block decomposition rule, which prevents
#   empty matches from composite AND-conditions (e.g. ["Biphenyl","Butadiyne"]) that match
#   zero edge building blocks, by splitting them into separate OR branches.
AGENT2_PROMPT_PATH = os.path.join(PROMPTS_DIR, "agent2_v4.1.md")

# Output directory
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")

# =============================================================================
# CAPABILITY MANIFEST (v3) - Describes what the system can evaluate
# Used for system metadata and experiment configuration.
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
    # hMOF gas uptakes — units are PER-FIELD per Wilmer source DOIs (verified 2026-05-26
    # from hmof_raw_cache.jsonl `adsorptionUnits` field). Pipeline performs NO unit
    # conversion (the build pipeline is pass-through).
    "h2_uptake_100bar_77K": {"display": "g/L", "type": "volumetric_mass"},
    "h2_uptake_2bar_77K": {"display": "g/L", "type": "volumetric_mass"},
    "ch4_uptake_35bar_298K": {"display": "cm³(STP)/cm³", "type": "volumetric"},
    "co2_uptake_2_5bar_298K": {"display": "mol/kg", "type": "gravimetric_molar"},
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
_PORMAKE_GPERL_ACTIVE = False        # g/L (set True at runtime by keyword detection)


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
    """Return the active Agent 1 prompt path (AGENT1_PROMPT_PATH)."""
    return AGENT1_PROMPT_PATH


# hMOF column mapping: hmof_index property names → sensitivity analyzer column names
HMOF_COLUMN_MAP = {
    "lcd": "di",  # largest cavity diameter
    "pld": "df",  # pore limiting diameter
    "surface_area_m2g": "sa",  # surface area (m2/cm3, Zeo++ ASA; key name is legacy)
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
FEEDBACK_SAMPLE_SIZE = 10  # Per-beam sample count (10 x 4 beams x 10 iters = 400)

# Sampling mode: stochastic (different samples each iteration)
STOCHASTIC_SAMPLING = True

# When True, exclude MOFs with missing density from volumetric feedback
# (prevents silent mol/kg → cm³(STP)/cm³ unit mismatch).
# Set False to revert to legacy behavior (keep rows with wrong units).
STRICT_DENSITY_CHECK = True

# Agent LLM Temperature Defaults (overridable via --agent1-temp / --agent2-temp CLI args)
AGENT1_TEMPERATURE = 0.0   # Deterministic hypothesis generation
AGENT2_TEMPERATURE = 0.0   # Deterministic constraint extraction (validated via temp ablation study)

# -----------------------------------------------------------------------------
# UNIVERSAL-LEVER TOGGLES (productionized research levers; default ON, reversible)
# -----------------------------------------------------------------------------
# Validated across the evaluation tasks (database mode, 5 replicates each). Each is firewall-clean
# (signals only from the agent's paid-for sampled candidates; identity-only keys;
# facts-only memory). Set False to fall back to legacy behavior bit-for-bit.
#   - STRATIFIED_SAMPLING: round-robin the feedback samples across the METALS present
#     in the matched beam, so rare-but-best metals (In/Ga) are not crowded out.
#     The only fully cross-app-validated universal lever.
#   - USE_MEMORY_LEDGER: prepend a FACTS-ONLY "design memory" (best-so-far + top-K
#     frontier + observational geometry) distilled from the agent's OWN sampled
#     beams. No exploit/prescriptive guidance (that variant hurt PORMAKE).
# soft-count (decoration committed SOFTLY: presence or <=2, never a high hard
# threshold) is delivered by the production agent1 prompt (AGENT1_PROMPT_PATH),
# not a code flag — revert by pointing AGENT1_PROMPT_PATH at the legacy prompt.
def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean toggle from the environment (override), else use the default.
    Accepts 0/1/true/false/yes/no/on/off (case-insensitive). Enables ops + test control
    without editing this file (e.g. LLM4MOF_STRATIFIED_SAMPLING=0 to disable for a run)."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


STRATIFIED_SAMPLING = _env_flag("LLM4MOF_STRATIFIED_SAMPLING", True)
USE_MEMORY_LEDGER = _env_flag("LLM4MOF_USE_MEMORY_LEDGER", True)

# Geometry-ranking margin for mof2zeo preranking (LIVE only). The ranking window is
# the agent's geometry_filter expanded by a per-descriptor margin so mof2zeo is not
# penalised by its own prediction error. Modes:
#   "mae"       -> model prediction error (~+/-0.75A di) — DEFAULT (the live weak-bridge fix)
#   "train_std" -> legacy 0.5 x data std (~+/-3.2A di) — old master behavior (now the worst)
#   "off"       -> rank against the strict agent window (no expansion)
# Default is "mae": the old "train_std" margin (~4x the model's real MAE) let mof2zeo ignore
# the agent's narrow window. The faithful local strict-pass-yield test ranks off > mae >
# train_std; we deploy "mae" (a small error-sized margin) rather than "off" because live
# PORMAKE-assembled MOFs have higher prediction error than the in-distribution valid set, so
# keeping a small cushion is safer. Override per-run with LLM4MOF_GEOM_MARGIN_MODE=train_std
# (reversible, no code edit).
GEOM_MARGIN_MODE = os.environ.get("LLM4MOF_GEOM_MARGIN_MODE", "mae").strip().lower()

# Conventional single-node single-edge scope is now BAKED INTO THE DATA: the PORMAKE markscheme
# DBs are pre-filtered to the core/mof2zeo/data whitelist (scripts/build_canonical_db.py) and the
# matchmaker proposes only from it — so there is no runtime on/off flag for it.


def is_stratified_sampling() -> bool:
    """True if metal-stratified feedback/simulation sampling is enabled."""
    return STRATIFIED_SAMPLING


def is_memory_ledger_enabled() -> bool:
    """True if the facts-only memory ledger is prepended to feedback."""
    return USE_MEMORY_LEDGER


# Agent 0 (the optional consultant-interview front-end) is out of scope for this release.

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
MOF2ZEO_CKPT_PATH = os.path.join(MOF2ZEO_DIR, "ckpt", "epoch=487-step=1039440.ckpt")

# Scaler files for inverse transform (mean/std from training data)
MOF2ZEO_SCALER_MEAN_PATH = os.path.join(MOF2ZEO_DIR, "data", "mean.csv")
MOF2ZEO_SCALER_STD_PATH = os.path.join(MOF2ZEO_DIR, "data", "std.csv")

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
# LIVE SIMULATION CONFIGURATION (live-simulation pipeline as feedback source)
# =============================================================================
# These settings control the live-simulation loop (run_live_experiment.py).
# The markscheme path (run_experiment.py) is unaffected.

_LIVE_SIM_ACTIVE = False  # Set True by run_live_experiment.py at startup

LIVE_SIM_N_PER_BEAM = 10           # Target successful simulations per beam (was 8)
LIVE_SIM_N_BEAMS = 4               # Z (full), A (chem-only), F (metal-only), Total (random)
LIVE_SIM_POOL_MULTIPLIER = 2       # Pool size = N_PER_BEAM * POOL_MULTIPLIER (was 3)
LIVE_SIM_POOL_MULTIPLIER_RANDOM = 2  # Pool size for Beam 4 (random baseline)
LIVE_SIM_MIN_SUCCESSES = 4         # Accept partial beam if >= this many successes

# Two-stage pipeline (--zeo mode): all beams run stage1 (pormake+lammps+zeo) in R1,
# then only successes proceed to RASPA in R2.
LIVE_SIM_Z_POOL_SIZE = 100         # Z beam stage1 pool size
LIVE_SIM_Z_RASPA_TOP = 10          # Top N from geometry-filter pass to run RASPA (stage2)
LIVE_SIM_AF_POOL_SIZE = 30         # A/F/total beam stage1 pool size (pormake+lammps+zeo only)
LIVE_SIM_CV_THRESHOLD = None      # Predicted CV (mof2zeo) upper limit (Å³); None = no filter
LIVE_SIM_MAX_COMBOS = 5000        # Max mof2zeo prediction candidates per beam (was 1000)

# LAMMPS topology blacklist (live simulation only, does not affect markscheme).
# Topologies listed here are excluded from all live sim beams before HPC submission.
# Add/remove entries to tune; set to empty set to disable: set()
LAMMPS_TOPOLOGY_BLACKLIST: set = set()

LIVE_SIM_RASPA_CYCLES = 5000       # Production: 5k cycles (reduced for speed; 5bar converges fast)
LIVE_SIM_RASPA_INIT_CYCLES = 5000  # Production: 5k init cycles (production default)
LIVE_SIM_RASPA_TEMPERATURE = 77.0  # K (hydrogen storage standard)
LIVE_SIM_RASPA_PRESSURE = 10000000.0  # Pa (~100 bar)

# Default adsorbate for the live simulation pipeline (H2 for backward compatibility)
LIVE_SIM_ADSORBATE = "h2"
LIVE_SIM_XE_MOLFRAC = 0.20   # Xe mole fraction for xekr mixture (Kr = 1 - xe_molfrac)

# Per-adsorbate simulation defaults — used by run_mof_sim.py and live_runner.py.
# Mirrors ADSORBATE_CONFIGS in core/simulation/gcmc/run_raspa.py; keep in sync.
LIVE_SIM_ADSORBATE_CONFIGS: dict = {
    "h2": {
        "forcefield": "UFF_H2",
        "molecule": "hydrogen",
        "temperature": 77.0,
        "pressure": 10000000.0,   # 100 bar
        "charge_method": "None",
        "mw_g_mol": 2.016,
        "xe_molfrac": None,
    },
    "ch4": {
        "forcefield": "UFF",
        "molecule": "CH4",
        "temperature": 298.0,
        "pressure": 250000.0,     # 2.5 bar
        "charge_method": "None",
        "mw_g_mol": 16.043,
        "xe_molfrac": None,
    },
    "co2": {
        "forcefield": "UFF",
        "molecule": "CO2",
        "temperature": 298.0,
        "pressure": 250000.0,     # 2.5 bar
        "charge_method": "Ewald",
        "mw_g_mol": 44.010,
        "xe_molfrac": None,
    },
    "xekr": {
        "forcefield": "UFF_XeKr",
        "molecule": None,         # 2-component mixture
        "temperature": 273.0,
        "pressure": 100000.0,     # 1 bar
        "charge_method": "None",
        "mw_g_mol": None,         # xe_mw=131.29, kr_mw=83.798 handled separately
        "xe_molfrac": 0.20,       # Xe 20% / Kr 80%
    },
}

LIVE_SIM_SKIP_LAMMPS = False       # LAMMPS enabled on HPC; local smoke tests override
LIVE_SIM_LAMMPS_TIMEOUT = 900      # 15 min cap per MOF
LIVE_SIM_RASPA_TIMEOUT = 7200      # 2 hours — empirical test showed 94 min max for 15k cycles
RASPA_MAX_LOADING_G_L = 71.0       # liquid H2 density at 20K (g/L) — physical upper bound for sanity gate

LIVE_SIM_MOF2ZEO_TOPN = 50         # mof2zeo top-N candidates per beam before sampling
LIVE_SIM_CACHE_DIR = os.path.join(BASE_DIR, "experiments")

LIVE_SIM_MAX_ITERATIONS = 10       # 10 iterations for production run

# HPC Configuration (PBS/Torque cluster — adapt these to your own environment)
HPC_HOST = "my-hpc"                 # SSH host alias (define it in ~/.ssh/config)
HPC_BASE_DIR = "~/llm4mof"          # Base directory on HPC
HPC_NODE_PROPERTY = "ac"            # optional PBS node property (cluster-specific)
HPC_SUBMIT_SCRIPT = "submit_iteration.sh"  # submit script on HPC (packed variant: submit_iteration_packed.sh)
HPC_SUBMIT_CMD = "qsub"             # batch submit command (set to your scheduler's)
HPC_STATUS_CMD = "qstat"            # job-status command (set to your scheduler's)
HPC_POLL_INTERVAL = 300             # 5 minutes between SSH polls
HPC_POLL_MAX_HOURS = 24             # Give up polling after this many hours
HPC_SSH_RETRIES = 10                # Retry SSH on connection failure
HPC_SSH_RETRY_DELAYS = [30, 60, 120, 120, 120, 120, 120, 120, 120]  # Backoff (seconds)


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
