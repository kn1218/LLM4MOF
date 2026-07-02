"""
Live Simulation Runner — generator-style MOF simulation with refill-on-failure.

Orchestrates the full pipeline per beam:
  matchmaker → SIM_SAFE filter → mof2zeo prefilter → ranked pool
  → PORMAKE build → LAMMPS optimize → RASPA3 GCMC → parse results

Each beam targets N successful simulations.  On failure, the next
mof2zeo-ranked candidate is pulled from the pool until N successes
are reached or the pool is exhausted.
"""

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.sim_safe_topologies import SIM_SAFE_TOPOS, filter_matchmaker_result
from core.filter_candidate import (
    MOFComponent, PredictedGeometry, RankedMOF, GeometryPredictor, MOFRanker,
    ComponentGenerator,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Result of one MOF simulation attempt."""
    topology: str
    node: str
    edge: str
    filename: str
    status: str          # "success" | "build_fail" | "lammps_fail" | "raspa_fail" | "stage1_done"
    predicted_geometry: Optional[Dict[str, float]] = None
    match_score: float = 0.0
    real_uptake: Optional[Dict[str, float]] = None   # parsed RASPA output
    real_geometry: Optional[Dict[str, float]] = None  # zeo++ computed geometry (if --zeo)
    cif_path: Optional[str] = None                   # stage1_only: CIF path for stage2 RASPA
    error_msg: str = ""
    wall_seconds: float = 0.0
    geo_filter_passed: bool = True   # Z beam stage2: False if promoted via fallback (did not pass strict filter)
    geo_filter_fail_reason: str = "" # Z beam fallback: comma-separated list of violated conditions e.g. "sa=1820<2500, vf=0.18<0.22"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeamResult:
    """Results for one beam."""
    beam_id: str         # "Z", "A", "F", "total"
    beam_label: str      # human-readable
    successes: List[SimResult] = field(default_factory=list)
    failures: List[SimResult] = field(default_factory=list)
    pool_size: int = 0
    target_n: int = 0
    matchmaker_diag: Dict = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return len(self.successes) >= self.target_n

    @property
    def is_acceptable(self) -> bool:
        return len(self.successes) >= config.LIVE_SIM_MIN_SUCCESSES


@dataclass
class LiveResults:
    """Full results of one live-simulation iteration."""
    beams: Dict[str, BeamResult] = field(default_factory=dict)
    n_real_simulations: int = 0
    n_failures: int = 0
    wall_clock_seconds: float = 0.0
    aborted_beams: List[str] = field(default_factory=list)
    geometry_aborted: bool = False          # True when Beam Z stage1: 0 pass geometry filter
    stage1_geometry_stats: Dict[str, Any] = field(default_factory=dict)  # actual geo distribution


# ---------------------------------------------------------------------------
# mof2zeo preranking: expand geometry_filter by 0.5 × train_std
# (applied only to mof2zeo ranking; Zeo++ strict filter uses original agent1 filter)
# ---------------------------------------------------------------------------

# Train-set std from mof2zeo training data (228K MOFs).
# Used to set a principled expansion margin so mof2zeo ranking isn't
# penalised by its own prediction error.
_MOF2ZEO_TRAIN_STD: Dict[str, float] = {
    "Di":      6.38,
    "Df":      6.03,
    "Dif":     6.43,
    "sa":      537.64,
    "vf":      0.19,
    "density": 0.33,
}
_MOF2ZEO_EXPAND_SIGMA: float = 0.5   # expand by 0.5 × train_std on each side

# Per-descriptor PREDICTION error (MAE) of the retrained 260614 mof2zeo model,
# measured on the held-out valid set 2026-06-15
# (held-out MAE evaluation of the mof2zeo model:
#  di MAE 0.80 / df 0.81 / dif 0.89 / sa 60 / vf 0.011 / density 0.018; p90 ~2x).
# These match the _MAE_SLACK in filter_candidate.py (within ~7%), so the prediction
# filter and this ranking expansion share one error-based margin. This is the principled
# replacement for _MOF2ZEO_TRAIN_STD (data spread), which was ~4x too loose (di ±3.2Å)
# and let mof2zeo ignore the agent's narrow geometry window. See config.GEOM_MARGIN_MODE.
_MOF2ZEO_PRED_ERR: Dict[str, float] = {
    "Di":      0.746,
    "Df":      0.751,
    "Dif":     0.828,
    "sa":      50.7,
    "vf":      0.0098,
    "density": 0.0152,
}


def _has_geometry_constraints(geometry_filter: Dict[str, Any]) -> bool:
    """Return True if geometry_filter has at least one non-null constraint."""
    return bool(geometry_filter) and any(v is not None for v in geometry_filter.values())


def _random_ranked_pool(components: List[MOFComponent]) -> List[RankedMOF]:
    """Return components as RankedMOF list with dummy geometry and score=0.

    Used when geometry filter is null — mof2zeo prediction is meaningless,
    and components are already shuffled by ComponentGenerator.
    """
    dummy = PredictedGeometry(sa=0.0, cv=0.0, density=0.0, vf=0.0, di=0.0, df=0.0, dif=0.0)
    return [
        RankedMOF(rank=i + 1, component=c, predicted_geometry=dummy,
                  match_score=0.0, geometry_match={})
        for i, c in enumerate(components)
    ]


def _apply_cv_filter(
    components: List[MOFComponent],
    predictions: List['PredictedGeometry'],
    cv_threshold: float,
    beam_id: str,
) -> tuple:
    """Filter components by predicted CV (mof2zeo). Returns (components, predictions)."""
    before = len(components)
    pairs = [(c, p) for c, p in zip(components, predictions) if p.cv <= cv_threshold]
    if pairs:
        components, predictions = zip(*pairs)
        components, predictions = list(components), list(predictions)
    else:
        components, predictions = [], []
    print(f"[Beam {beam_id}] CV filter (pred_cv ≤ {cv_threshold:,.0f}): {len(components)}/{before} passed")
    return components, predictions



def _margin_table() -> Dict[str, float]:
    """Per-descriptor expansion margin for mof2zeo preranking, per config.GEOM_MARGIN_MODE.

    - "mae"       : model prediction error (_MOF2ZEO_PRED_ERR) — respects the agent window  [default]
    - "train_std" : legacy 0.5 × data std (_MOF2ZEO_TRAIN_STD) — ~4× looser (pre-2026-06-15)
    - "off"       : no expansion (rank against the strict agent window)
    """
    mode = getattr(config, "GEOM_MARGIN_MODE", "mae")
    if mode == "off":
        return {}
    if mode == "train_std":
        return {prop: _MOF2ZEO_EXPAND_SIGMA * std for prop, std in _MOF2ZEO_TRAIN_STD.items()}
    # "mae" (default — the live weak-bridge fix)
    return dict(_MOF2ZEO_PRED_ERR)


def _expand_geometry_filter(gf: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of geometry_filter with bounds relaxed by a per-descriptor margin.

    Only used for mof2zeo preranking.  The original filter (agent1 output) is kept
    intact and applied to real Zeo++ geometry later.  The margin source is selected
    by config.GEOM_MARGIN_MODE (see _margin_table): "mae" (default) uses the model's
    measured prediction error; "train_std" is the legacy data-spread margin; "off"
    ranks against the strict agent window.

    Keys follow the agent2 convention: target_{Prop}_{min,max} where Prop is
    Di, Df, Dif, sa, vf, density.  min bounds are lowered (floor at 0);
    max bounds are raised.  Missing bounds are not synthesised.
    """
    if not gf:
        return gf
    expanded = dict(gf)
    for prop, margin in _margin_table().items():
        min_key = f"target_{prop}_min"
        max_key = f"target_{prop}_max"
        if min_key in expanded and expanded[min_key] is not None:
            expanded[min_key] = max(0.0, expanded[min_key] - margin)
        if max_key in expanded and expanded[max_key] is not None:
            expanded[max_key] = expanded[max_key] + margin
    return expanded


# ---------------------------------------------------------------------------
# Geometry match scoring (range-aware, shared by local and HPC paths)
# ---------------------------------------------------------------------------

def compute_geometry_match_score(real_geom: dict, geometry_filter: dict) -> float:
    """
    Score how well real_geom matches the geometry_filter constraints.

    Range-aware: score = 1 at midpoint of [min, max], decays linearly to 0
    at and beyond the boundary.  For min-only constraints (no max), score = 1
    if value >= min, decaying linearly below.  Penalises values exceeding max.

    Returns a float in [0, 1]; higher = better match.
    """
    if not geometry_filter or not real_geom:
        return 0.0

    # (min_key, max_key, geom_key, weight)
    specs = [
        ("target_Di_min",      "target_Di_max",      "di",      1.0),
        ("target_Df_min",      "target_Df_max",      "df",      2.0),  # most important for H2@5bar
        ("target_sa_min",      "target_sa_max",      "sa",      1.5),
        ("target_vf_min",      "target_vf_max",      "vf",      1.0),
        ("target_density_min", "target_density_max", "density", 1.0),
        ("target_dif_min",     "target_dif_max",     "dif",     0.5),
    ]
    total_weight = 0.0
    weighted_sum = 0.0
    for min_key, max_key, geom_key, weight in specs:
        lo = geometry_filter.get(min_key)
        hi = geometry_filter.get(max_key)
        val = real_geom.get(geom_key)
        if lo is None or val is None:
            continue
        val = float(val)
        lo = float(lo)
        if hi is not None:
            hi = float(hi)
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            if lo <= val <= hi:
                s = 1.0 - abs(val - mid) / half
            elif val < lo:
                s = max(0.0, 1.0 - (lo - val) / half)
            else:  # val > hi
                s = max(0.0, 1.0 - (val - hi) / half)
        else:
            # min-only: full credit at threshold, linear decay below, capped at 1.0 above
            s = min(val / lo, 1.0) if lo > 0 else 1.0
            s = max(0.0, s)
        total_weight += weight
        weighted_sum += weight * s

    if total_weight == 0.0:
        return 0.5  # No criteria to evaluate — neutral score

    return weighted_sum / total_weight


def passes_geometry_filter(geom: dict, geometry_filter: dict) -> bool:
    """Return True if zeo++ geometry satisfies all non-null geometry_filter constraints.

    geometry_filter keys: target_Di_min/max, target_Df_min/max,
    target_sa_min/max, target_vf_min/max, target_density_min/max, target_dif_min/max.
    geom keys: di, df, sa (m²/cm³), vf, density, dif, cv.
    """
    if not geometry_filter:
        return True
    checks = [
        ("target_Di_min",      "di",      "ge"),
        ("target_Di_max",      "di",      "le"),
        ("target_Df_min",      "df",      "ge"),
        ("target_Df_max",      "df",      "le"),
        ("target_dif_min",     "dif",     "ge"),
        ("target_dif_max",     "dif",     "le"),
        ("target_sa_min",      "sa",      "ge"),
        ("target_sa_max",      "sa",      "le"),
        ("target_vf_min",      "vf",      "ge"),
        ("target_vf_max",      "vf",      "le"),
        ("target_density_min", "density", "ge"),
        ("target_density_max", "density", "le"),
    ]
    for fkey, gkey, op in checks:
        threshold = geometry_filter.get(fkey)
        value = geom.get(gkey)
        if threshold is None or value is None:
            continue
        if op == "ge" and float(value) < float(threshold):
            return False
        if op == "le" and float(value) > float(threshold):
            return False
    return True


def geometry_fail_reason(geom: dict, geometry_filter: dict) -> str:
    """Return a human-readable string of violated geometry_filter conditions.

    E.g. "sa=1820<2500, vf=0.18<0.22". Returns "" if all conditions pass or filter is empty.
    """
    if not geometry_filter or not geom:
        return ""
    checks = [
        ("target_Di_min",      "di",      "ge"),
        ("target_Di_max",      "di",      "le"),
        ("target_Df_min",      "df",      "ge"),
        ("target_Df_max",      "df",      "le"),
        ("target_dif_min",     "dif",     "ge"),
        ("target_dif_max",     "dif",     "le"),
        ("target_sa_min",      "sa",      "ge"),
        ("target_sa_max",      "sa",      "le"),
        ("target_vf_min",      "vf",      "ge"),
        ("target_vf_max",      "vf",      "le"),
        ("target_density_min", "density", "ge"),
        ("target_density_max", "density", "le"),
    ]
    reasons = []
    for fkey, gkey, op in checks:
        threshold = geometry_filter.get(fkey)
        value = geom.get(gkey)
        if threshold is None or value is None:
            continue
        if op == "ge" and float(value) < float(threshold):
            reasons.append(f"{gkey}={float(value):.4g}<{float(threshold):.4g}")
        elif op == "le" and float(value) > float(threshold):
            reasons.append(f"{gkey}={float(value):.4g}>{float(threshold):.4g}")
    return ", ".join(reasons)


# ---------------------------------------------------------------------------
# Simulation cache (persists successes AND failures across iterations)
# ---------------------------------------------------------------------------

class SimCache:
    """Append-only JSONL cache keyed by (topology, node, edge)."""

    def __init__(self, cache_path: str):
        self._path = cache_path
        self._cache: Dict[Tuple[str, str, str], SimResult] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    key = (d["topology"], d["node"], d["edge"])
                    self._cache[key] = SimResult(**{
                        k: d.get(k) for k in SimResult.__dataclass_fields__
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[SimCache] Loaded {len(self._cache)} entries from {self._path}")

    def get(self, topo: str, node: str, edge: str) -> Optional[SimResult]:
        return self._cache.get((topo, node, edge))

    def put(self, result: SimResult) -> None:
        key = (result.topology, result.node, result.edge)
        self._cache[key] = result
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Single-MOF simulation pipeline
# ---------------------------------------------------------------------------

def _simulate_one_mof(
    component: MOFComponent,
    work_dir: str,
    predicted_geom: Optional[PredictedGeometry] = None,
    match_score: float = 0.0,
    use_zeo: bool = False,
    zeopp_bin: Optional[str] = None,
    apply_geo_filter: bool = False,
    geometry_filter: Optional[Dict[str, Any]] = None,
    pipeline: str = "full",
    stage2_cif_path: Optional[str] = None,
    stage2_real_geometry: Optional[Dict[str, float]] = None,
) -> SimResult:
    """
    Run MOF simulation pipeline with pipeline-mode support.

    pipeline:
      "full"        — pormake → lammps → [zeo++] → raspa (existing behavior)
      "stage1_only" — pormake → lammps → zeo++ only; returns status="stage1_done" with cif_path
      "stage2_only" — raspa only; uses stage2_cif_path + stage2_real_geometry from stage1

    Returns a SimResult with status and parsed uptake (if successful).
    """
    t0 = time.time()
    filename = component.filename
    pred_dict = predicted_geom.to_dict() if predicted_geom else None

    # -----------------------------------------------------------------------
    # Stage2-only: skip pormake/lammps/zeo, run RASPA with existing CIF
    # -----------------------------------------------------------------------
    if pipeline == "stage2_only":
        if not stage2_cif_path or not os.path.isfile(stage2_cif_path):
            return SimResult(
                topology=component.topology, node=component.node,
                edge=component.edge, filename=filename,
                status="stage2_fail", predicted_geometry=pred_dict,
                match_score=match_score,
                error_msg=f"CIF not found for stage2: {stage2_cif_path}",
                wall_seconds=time.time() - t0,
            )

        opt_cif_dir = os.path.dirname(stage2_cif_path)
        real_geometry = stage2_real_geometry

        try:
            from core.simulation.gcmc.run_raspa import (
                create_raspa_input, run_simulation_background,
                FORCEFIELD_BASE_DIR, DEFAULT_RASPA3,
            )
            from core.simulation.gcmc.raspa_utils import parse_output, parse_output_mixture

            _ads = getattr(config, "LIVE_SIM_ADSORBATE", "h2")
            _ads_cfgs = getattr(config, "LIVE_SIM_ADSORBATE_CONFIGS", {})
            _ads_cfg = _ads_cfgs.get(_ads, _ads_cfgs.get("h2", {}))
            _ff_dir = os.path.join(FORCEFIELD_BASE_DIR, _ads_cfg.get("forcefield", "UFF_H2"))
            _xe_molfrac = getattr(config, "LIVE_SIM_XE_MOLFRAC", _ads_cfg.get("xe_molfrac") or 0.20)

            raspa_out = os.path.join(work_dir, "raspa_output")
            mof_dict = {"filename": filename}
            params = {
                "adsorbate": _ads,
                "cycles": config.LIVE_SIM_RASPA_CYCLES,
                "init_cycles": config.LIVE_SIM_RASPA_INIT_CYCLES,
                "temperature": config.LIVE_SIM_RASPA_TEMPERATURE,
                "pressure": config.LIVE_SIM_RASPA_PRESSURE,
                "forcefield_dir": _ff_dir,
                "xe_molfrac": _xe_molfrac,
            }
            if not DEFAULT_RASPA3:
                return SimResult(
                    topology=component.topology, node=component.node,
                    edge=component.edge, filename=filename,
                    status="raspa_fail", predicted_geometry=pred_dict,
                    match_score=match_score, real_geometry=real_geometry,
                    error_msg="raspa3 binary not found",
                    wall_seconds=time.time() - t0,
                )
            input_file = create_raspa_input(mof_dict, opt_cif_dir, raspa_out, params)
            ok = run_simulation_background(
                input_file, opt_cif_dir, raspa_out,
                _ff_dir, filename, DEFAULT_RASPA3,
                timeout=config.LIVE_SIM_RASPA_TIMEOUT,
            )
            if not ok:
                return SimResult(
                    topology=component.topology, node=component.node,
                    edge=component.edge, filename=filename,
                    status="raspa_fail", predicted_geometry=pred_dict,
                    match_score=match_score, real_geometry=real_geometry,
                    error_msg="RASPA3 simulation failed",
                    wall_seconds=time.time() - t0,
                )
            mof_output_dir = os.path.join(raspa_out, filename)
            cif_for_density = stage2_cif_path if stage2_cif_path and os.path.isfile(stage2_cif_path) else None
            if _ads == "xekr":
                parsed = parse_output_mixture(mof_output_dir, xe_molfrac=_xe_molfrac, cif_path=cif_for_density)
                if not parsed or parsed.get("xe_loading_mol_kg") is None:
                    return SimResult(
                        topology=component.topology, node=component.node,
                        edge=component.edge, filename=filename,
                        status="raspa_fail", predicted_geometry=pred_dict,
                        match_score=match_score, real_geometry=real_geometry,
                        error_msg="RASPA3 Xe/Kr output parsing failed",
                        wall_seconds=time.time() - t0,
                    )
            else:
                parsed = parse_output(mof_output_dir, adsorbate_mw_g_mol=_ads_cfg.get("mw_g_mol") or 2.016)
                if not parsed or not parsed.get("loading_mol_kg"):
                    return SimResult(
                        topology=component.topology, node=component.node,
                        edge=component.edge, filename=filename,
                        status="raspa_fail", predicted_geometry=pred_dict,
                        match_score=match_score, real_geometry=real_geometry,
                        error_msg="RASPA3 output parsing failed",
                        wall_seconds=time.time() - t0,
                    )
                mol_kg = float(parsed.get("loading_mol_kg", 0.0) or 0.0)
                zeo_density = float((real_geometry or {}).get("density", 0.0) or 0.0)
                pred_density = float((pred_dict or {}).get("density", 0.0) or 0.0)
                density_for_gl = zeo_density if zeo_density > 0 else pred_density
                mw = _ads_cfg.get("mw_g_mol") or 2.016
                parsed["loading_g_L"] = round(mol_kg * density_for_gl * mw, 6)
                # Physical sanity gate (H2 only)
                if _ads == "h2" and parsed["loading_g_L"] > config.RASPA_MAX_LOADING_G_L:
                    return SimResult(
                        topology=component.topology, node=component.node,
                        edge=component.edge, filename=filename,
                        status="raspa_fail", predicted_geometry=pred_dict,
                        match_score=match_score, real_uptake=parsed,
                        real_geometry=real_geometry,
                        error_msg=(f"loading_g_L={parsed['loading_g_L']:.1f} exceeds physical "
                                   f"limit ({config.RASPA_MAX_LOADING_G_L} g/L, liquid H2 at 20K)"),
                        wall_seconds=time.time() - t0,
                    )
            return SimResult(
                topology=component.topology, node=component.node,
                edge=component.edge, filename=filename,
                status="success", predicted_geometry=pred_dict,
                match_score=match_score, real_uptake=parsed,
                real_geometry=real_geometry,
                wall_seconds=time.time() - t0,
            )
        except Exception as e:
            return SimResult(
                topology=component.topology, node=component.node,
                edge=component.edge, filename=filename,
                status="raspa_fail", predicted_geometry=pred_dict,
                match_score=match_score, real_geometry=real_geometry,
                error_msg=str(e)[:300],
                wall_seconds=time.time() - t0,
            )

    # -----------------------------------------------------------------------
    # Full or stage1_only: pormake → lammps → [zeo++] [→ raspa]
    # -----------------------------------------------------------------------
    cif_dir = os.path.join(work_dir, "cif")
    os.makedirs(cif_dir, exist_ok=True)

    # --- Stage 1: PORMAKE build ---
    try:
        from core.simulation.generate_mofs import generate_mof
        import pormake as pm

        database = pm.Database()
        builder = pm.Builder()
        mof_info = {
            "filename": filename,
            "topology": component.topology,
            "node": component.node,
            "edge": component.edge,
        }
        ok = generate_mof(mof_info, database, builder, cif_dir)
        if not ok:
            return SimResult(
                topology=component.topology, node=component.node,
                edge=component.edge, filename=filename,
                status="build_fail", predicted_geometry=pred_dict,
                match_score=match_score, error_msg="PORMAKE build failed",
                wall_seconds=time.time() - t0,
            )
    except Exception as e:
        return SimResult(
            topology=component.topology, node=component.node,
            edge=component.edge, filename=filename,
            status="build_fail", predicted_geometry=pred_dict,
            match_score=match_score, error_msg=str(e)[:300],
            wall_seconds=time.time() - t0,
        )

    cif_path = os.path.join(cif_dir, f"{filename}.cif")

    # --- Stage 2: LAMMPS optimize (optional) ---
    # We call each step individually instead of run_pipeline() to avoid
    # the blocking wait_for_optimizations() loop that hangs on failure.
    active_cif_path = cif_path  # updated to optimized CIF on success
    if not config.LIVE_SIM_SKIP_LAMMPS:
        _lammps_fail_msg = None
        try:
            from core.simulation.opt.optimize import (
                run_lammps_interface, make_optimization_input,
                run_lammps_optimization, convert_lammps_to_cif,
            )
            lammps_out = Path(os.path.join(work_dir, "lammps_output"))
            data_dir = lammps_out / "lammps_data"
            data_dir.mkdir(exist_ok=True, parents=True)

            # Step 1: lammps-interface (CIF → LAMMPS data file)
            cif_path_obj = Path(cif_path)
            if_ok = run_lammps_interface(cif_path_obj, data_dir)
            if not if_ok:
                _lammps_fail_msg = "lammps-interface failed"
            else:
                # Step 2: Create optimization input + run LAMMPS
                created = make_optimization_input(data_dir, lammps_out)
                if not created:
                    _lammps_fail_msg = "no input files created"
                else:
                    opt_ok = run_lammps_optimization(created[0], lammps_out)
                    if not opt_ok:
                        _lammps_fail_msg = "optimization failed"
                    else:
                        # Step 3: Convert optimized data → CIF
                        opt_data_dir = lammps_out / "opt_lammps_data"
                        opt_cif_out = lammps_out / "optimized_cifs"
                        opt_cif_out.mkdir(exist_ok=True, parents=True)
                        lammps_files = list(opt_data_dir.glob("*_opt.lammps-data"))
                        if not lammps_files:
                            _lammps_fail_msg = "no optimized data produced"
                        else:
                            name = lammps_files[0].stem.replace("data.", "").replace("_opt", "")
                            convert_lammps_to_cif(lammps_files[0], opt_cif_out / f"{name}.cif")
                            opt_cif_files = [f for f in opt_cif_out.iterdir() if f.name.endswith(".cif")]
                            if opt_cif_files:
                                active_cif_path = str(opt_cif_files[0])
                                print(f"   [LAMMPS] Using optimized CIF")
                            else:
                                _lammps_fail_msg = "CIF conversion failed"
        except Exception as e:
            _lammps_fail_msg = str(e)[:200]

        if _lammps_fail_msg:
            print(f"   [LAMMPS] Failed ({_lammps_fail_msg})")
            return SimResult(
                topology=component.topology, node=component.node,
                edge=component.edge, filename=filename,
                status="lammps_fail", predicted_geometry=pred_dict,
                match_score=match_score, error_msg=_lammps_fail_msg,
                wall_seconds=time.time() - t0,
            )

    # --- Stage 3: Zeo++ geometry (runs BEFORE RASPA; always for stage1_only) ---
    zeo_cif_path = active_cif_path
    real_geometry = None
    run_zeo_flag = use_zeo or (pipeline == "stage1_only")
    if run_zeo_flag and zeopp_bin:
        try:
            from core.simulation.zeo.run_zeo import run_zeo
            real_geometry = run_zeo(zeo_cif_path, zeopp_bin)
        except Exception as e:
            print(f"   [Zeo++] Error for {filename}: {e}")
            real_geometry = None

        if real_geometry is None:
            return SimResult(
                topology=component.topology, node=component.node,
                edge=component.edge, filename=filename,
                status="zeo_fail", predicted_geometry=pred_dict,
                match_score=match_score,
                error_msg="Zeo++ failed or returned incomplete geometry",
                wall_seconds=time.time() - t0,
            )

        if apply_geo_filter and geometry_filter:
            if not passes_geometry_filter(real_geometry, geometry_filter):
                print(f"   [ZEO FILTER] {filename}: geometry not satisfied — proceeding to RASPA anyway")

    # --- Stage1-only: return after zeo, before RASPA ---
    if pipeline == "stage1_only":
        df_val = real_geometry.get("df") if real_geometry else "N/A"
        print(f"   [STAGE1 DONE] {filename}: df={df_val}")
        return SimResult(
            topology=component.topology, node=component.node,
            edge=component.edge, filename=filename,
            status="stage1_done", predicted_geometry=pred_dict,
            match_score=match_score, real_geometry=real_geometry,
            cif_path=zeo_cif_path,  # LAMMPS-optimized CIF for stage2 RASPA
            wall_seconds=time.time() - t0,
        )

    # --- Stage 4: RASPA3 GCMC ---
    try:
        from core.simulation.gcmc.run_raspa import (
            create_raspa_input, run_simulation_background,
            FORCEFIELD_BASE_DIR, DEFAULT_RASPA3,
        )
        from core.simulation.gcmc.raspa_utils import parse_output, parse_output_mixture

        _ads = getattr(config, "LIVE_SIM_ADSORBATE", "h2")
        _ads_cfgs = getattr(config, "LIVE_SIM_ADSORBATE_CONFIGS", {})
        _ads_cfg = _ads_cfgs.get(_ads, _ads_cfgs.get("h2", {}))
        _ff_dir = os.path.join(FORCEFIELD_BASE_DIR, _ads_cfg.get("forcefield", "UFF_H2"))
        _xe_molfrac = getattr(config, "LIVE_SIM_XE_MOLFRAC", _ads_cfg.get("xe_molfrac") or 0.20)

        raspa_out = os.path.join(work_dir, "raspa_output")
        mof_dict = {"filename": filename}
        params = {
            "adsorbate": _ads,
            "cycles": config.LIVE_SIM_RASPA_CYCLES,
            "init_cycles": config.LIVE_SIM_RASPA_INIT_CYCLES,
            "temperature": config.LIVE_SIM_RASPA_TEMPERATURE,
            "pressure": config.LIVE_SIM_RASPA_PRESSURE,
            "forcefield_dir": _ff_dir,
            "xe_molfrac": _xe_molfrac,
        }

        input_file = create_raspa_input(mof_dict, opt_cif_dir, raspa_out, params)

        raspa3_bin = DEFAULT_RASPA3
        if not raspa3_bin:
            return SimResult(
                topology=component.topology, node=component.node,
                edge=component.edge, filename=filename,
                status="raspa_fail", predicted_geometry=pred_dict,
                match_score=match_score,
                error_msg="raspa3 binary not found",
                wall_seconds=time.time() - t0,
            )

        ok = run_simulation_background(
            input_file, opt_cif_dir, raspa_out,
            _ff_dir, filename, raspa3_bin,
            timeout=config.LIVE_SIM_RASPA_TIMEOUT,
        )

        if not ok:
            return SimResult(
                topology=component.topology, node=component.node,
                edge=component.edge, filename=filename,
                status="raspa_fail", predicted_geometry=pred_dict,
                match_score=match_score,
                error_msg="RASPA3 simulation failed (non-zero exit or timeout)",
                wall_seconds=time.time() - t0,
            )

        # Parse output
        mof_output_dir = os.path.join(raspa_out, filename)
        if _ads == "xekr":
            cif_for_density = cif_path if os.path.isfile(cif_path) else None
            parsed = parse_output_mixture(mof_output_dir, xe_molfrac=_xe_molfrac, cif_path=cif_for_density)
            if not parsed or parsed.get("xe_loading_mol_kg") is None:
                return SimResult(
                    topology=component.topology, node=component.node,
                    edge=component.edge, filename=filename,
                    status="raspa_fail", predicted_geometry=pred_dict,
                    match_score=match_score,
                    error_msg="RASPA3 Xe/Kr output parsing failed",
                    wall_seconds=time.time() - t0,
                )
        else:
            mw = _ads_cfg.get("mw_g_mol") or 2.016
            parsed = parse_output(mof_output_dir, adsorbate_mw_g_mol=mw)
            if not parsed or not parsed.get("loading_mol_kg"):
                return SimResult(
                    topology=component.topology, node=component.node,
                    edge=component.edge, filename=filename,
                    status="raspa_fail", predicted_geometry=pred_dict,
                    match_score=match_score,
                    error_msg="RASPA3 output parsing failed (no loading data)",
                    wall_seconds=time.time() - t0,
                )
            mol_kg = float(parsed.get("loading_mol_kg", 0.0) or 0.0)
            # Compute loading_g_L: prefer zeo++ density, fall back to predicted
            zeo_density = float((real_geometry or {}).get("density", 0.0) or 0.0)
            pred_density = float((pred_dict or {}).get("density", 0.0) or 0.0)
            density_for_gl = zeo_density if zeo_density > 0 else pred_density
            parsed["loading_g_L"] = round(mol_kg * density_for_gl * mw, 6)
            # Physical sanity gate (H2 only)
            if _ads == "h2" and parsed["loading_g_L"] > config.RASPA_MAX_LOADING_G_L:
                return SimResult(
                    topology=component.topology, node=component.node,
                    edge=component.edge, filename=filename,
                    status="raspa_fail", predicted_geometry=pred_dict,
                    match_score=match_score, real_uptake=parsed,
                    real_geometry=real_geometry,
                    error_msg=(f"loading_g_L={parsed['loading_g_L']:.1f} exceeds physical "
                               f"limit ({config.RASPA_MAX_LOADING_G_L} g/L, liquid H2 at 20K)"),
                    wall_seconds=time.time() - t0,
                )

        return SimResult(
            topology=component.topology, node=component.node,
            edge=component.edge, filename=filename,
            status="success", predicted_geometry=pred_dict,
            match_score=match_score, real_uptake=parsed,
            real_geometry=real_geometry,
            wall_seconds=time.time() - t0,
        )

    except Exception as e:
        return SimResult(
            topology=component.topology, node=component.node,
            edge=component.edge, filename=filename,
            status="raspa_fail", predicted_geometry=pred_dict,
            match_score=match_score, error_msg=str(e)[:300],
            wall_seconds=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Beam construction helpers
# ---------------------------------------------------------------------------

def _build_beam_specs(full_specs: Dict[str, Any], beam_id: str) -> Dict[str, Any]:
    """
    Construct beam-specific specs by selectively dropping constraints.

    Beam Z: full specs (chemistry + geometry)
    Beam A: chemistry only (drop geometry_filter)
    Beam F: metal only (drop linker chemistry constraints)
    Beam total: None — handled separately via random sampling
    """
    import copy
    specs = copy.deepcopy(full_specs)

    if beam_id == "Z":
        return specs

    if beam_id == "A":
        # Drop geometry constraints
        specs.pop("geometry_filter", None)
        return specs

    if beam_id == "F":
        # Drop linker-specific constraints (keep metals and geometry)
        if "linker_query" in specs:
            specs["linker_query"].pop("functional_groups", None)
            specs["linker_query"].pop("linker_branches", None)
            specs["linker_query"].pop("abstract_features", None)
        if "global_requirements" in specs:
            specs.pop("global_requirements", None)
        return specs

    return specs


def _build_random_pool(
    n_candidates: int,
) -> List[MOFComponent]:
    """
    Build a random pool from SIM_SAFE_TOPOS × all nodes × all edges.
    Used for Beam 4 (global baseline).
    """
    from core.filter_candidate import ComponentGenerator

    gen = ComponentGenerator()
    safe_topos = list(SIM_SAFE_TOPOS)

    components = []
    attempts = 0
    max_attempts = max(n_candidates * 50, 500)

    while len(components) < n_candidates and attempts < max_attempts:
        topo = random.choice(safe_topos)
        node = random.choice(gen._node_list)
        edge = random.choice(gen._edge_list)

        # Validate connectivity
        topo_cn = gen._topo_cn.get(topo, 0)
        node_cn = gen._node_cn.get(node, 0)
        edge_cn = gen._edge_cn.get(edge, 0)

        if topo_cn == node_cn and edge_cn == 2:
            components.append(MOFComponent(topology=topo, node=node, edge=edge))

        attempts += 1

    return components


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_live_iteration(
    specs: Dict[str, Any],
    matchmaker,
    iteration: int,
    run_dir: str,
    sim_cache: SimCache,
    n_per_beam: int = config.LIVE_SIM_N_PER_BEAM,
    use_zeo: bool = False,
) -> LiveResults:
    """
    Run one live-simulation iteration across all 4 beams.

    Args:
        specs: Agent 2's constraint dict
        matchmaker: Matchmaker instance (already initialized)
        iteration: Current iteration number (1-based)
        run_dir: Directory for this experiment run
        sim_cache: Persistent simulation cache
        n_per_beam: Target successful simulations per beam

    Returns:
        LiveResults with all beam outcomes
    """
    t_start = time.time()
    live_results = LiveResults()

    _ads = getattr(config, "LIVE_SIM_ADSORBATE", "h2")

    predictor = GeometryPredictor()
    ranker = MOFRanker()
    gen = ComponentGenerator()

    target_geometry = specs.get("geometry_filter", {})

    # Resolve Zeo++ binary once per iteration (find or compile)
    zeopp_bin: Optional[str] = None
    if use_zeo:
        from core.simulation.zeo.install import ensure_zeopp
        zeopp_bin = ensure_zeopp()
        print(f"[Live Runner] Zeo++ enabled: {zeopp_bin}")

    # Two-stage mode for Beam Z: pormake+lammps+zeo for a large pool → filter → RASPA
    two_stage_z = use_zeo and bool(target_geometry)

    beam_configs = [
        ("Z", "Full Hypothesis (Chemistry + Geometry)", config.LIVE_SIM_POOL_MULTIPLIER),
        ("A", "Chemistry Only (no geometry gate)", config.LIVE_SIM_POOL_MULTIPLIER),
        ("F", "Metal Only (no linker constraints)", config.LIVE_SIM_POOL_MULTIPLIER),
        ("total", "Random Baseline (global)", config.LIVE_SIM_POOL_MULTIPLIER_RANDOM),
    ]

    for beam_id, beam_label, pool_mult in beam_configs:
        print(f"\n{'='*60}")
        print(f"[Beam {beam_id}] {beam_label}")
        print(f"{'='*60}")

        if beam_id == "Z" and two_stage_z:
            pool_size = config.LIVE_SIM_Z_POOL_SIZE
        elif beam_id != "Z" and use_zeo:
            pool_size = config.LIVE_SIM_AF_POOL_SIZE
        elif beam_id == "Z" and use_zeo:
            pool_size = config.LIVE_SIM_AF_POOL_SIZE
        else:
            pool_size = n_per_beam * pool_mult
        iter_dir = os.path.join(run_dir, f"iter_{iteration}", f"beam_{beam_id}")
        os.makedirs(iter_dir, exist_ok=True)

        # --- Build ranked candidate pool ---
        beam_mm_diag = {}
        beam_preferred = {}  # preferred_features for soft ranking bonus
        if beam_id == "total":
            # Random baseline: skip matchmaker, sample from full space
            # Oversample when CV threshold is active so filtering still yields pool_size candidates
            _cv_thr = config.LIVE_SIM_CV_THRESHOLD
            _raw_size = 1000 if _cv_thr is not None else pool_size
            components = _build_random_pool(_raw_size)
            print(f"[Beam total] Random pool: {len(components)} candidates")
        else:
            # Run matchmaker with beam-specific specs
            beam_specs = _build_beam_specs(specs, beam_id)
            mm_result = matchmaker.smart_matchmaker_single_node(beam_specs)
            beam_mm_diag = mm_result.get("diagnostics", {})  # capture before filter
            beam_preferred = mm_result.get("preferred_features", {})

            if mm_result.get("status") == "error":
                print(f"[Beam {beam_id}] Matchmaker error: {mm_result.get('message')}")
                live_results.beams[beam_id] = BeamResult(
                    beam_id=beam_id, beam_label=beam_label,
                    pool_size=0, target_n=n_per_beam,
                    matchmaker_diag=beam_mm_diag,
                )
                live_results.aborted_beams.append(beam_id)
                continue

            # Apply SIM_SAFE filter
            mm_result = filter_matchmaker_result(mm_result)

            if not mm_result.get("topology"):
                print(f"[Beam {beam_id}] No simulation-safe topologies after filter")
                live_results.beams[beam_id] = BeamResult(
                    beam_id=beam_id, beam_label=beam_label,
                    pool_size=0, target_n=n_per_beam,
                    matchmaker_diag=beam_mm_diag,
                )
                live_results.aborted_beams.append(beam_id)
                continue

            # Generate combinations
            try:
                components = gen.generate_from_matchmaker(mm_result, max_combos=config.LIVE_SIM_MAX_COMBOS)
            except ValueError as e:
                print(f"[Beam {beam_id}] ComponentGenerator error: {e}")
                live_results.beams[beam_id] = BeamResult(
                    beam_id=beam_id, beam_label=beam_label,
                    pool_size=0, target_n=n_per_beam,
                    matchmaker_diag=beam_mm_diag,
                )
                live_results.aborted_beams.append(beam_id)
                continue

        # --- mof2zeo predict + rank ---
        # Z beam (with geometry filter): predict → CV filter → rank.
        # A/F/total: predict only when CV threshold is set (for CV filter); then random rank.
        cv_threshold = config.LIVE_SIM_CV_THRESHOLD
        if beam_id == "Z" and _has_geometry_constraints(target_geometry):
            print(f"[Beam {beam_id}] Predicting geometry for {len(components)} candidates (mof2zeo)...")
            predictions = predictor.predict_batch(components)
            if cv_threshold is not None:
                components, predictions = _apply_cv_filter(components, predictions, cv_threshold, beam_id)
            ranked = ranker.rank(
                components, predictions, _expand_geometry_filter(target_geometry),
                preferred_features=beam_preferred or None,
                bb_lookup=matchmaker.bb_lookup if beam_preferred else None,
            ) if components else []
        else:
            if cv_threshold is not None:
                print(f"[Beam {beam_id}] Predicting geometry for CV filter ({len(components)} candidates)...")
                predictions = predictor.predict_batch(components)
                components, _ = _apply_cv_filter(components, predictions, cv_threshold, beam_id)
            else:
                print(f"[Beam {beam_id}] No geometry filter → skipping mof2zeo (random pool)")
            ranked = _random_ranked_pool(components)

        # Take top pool_size candidates. When metal-stratified sampling is enabled, pick a
        # metal-diverse subset (rank-order preserved within each metal) so rare-but-best metals
        # (In/Ga) enter the expensive simulation pool instead of being crowded out by pure
        # geometry-match rank. Firewall: identity only; fail-safe -> top-N. Gated by config.
        if config.is_stratified_sampling():
            from core import sampling_strata
            ranked_pool = sampling_strata.stratified_rank_select(ranked, pool_size)
        else:
            ranked_pool = ranked[:pool_size]
        if not ranked_pool:
            print(f"[Beam {beam_id}] Pool empty after ranking — skipping")
            live_results.beams[beam_id] = BeamResult(
                beam_id=beam_id, beam_label=beam_label,
                pool_size=0, target_n=n_per_beam,
            )
            live_results.aborted_beams.append(beam_id)
            continue
        print(f"[Beam {beam_id}] Pool: {len(ranked_pool)} candidates "
              f"(top score={ranked_pool[0].match_score:.3f})")

        # --- Simulation ---
        beam_result = BeamResult(
            beam_id=beam_id, beam_label=beam_label,
            pool_size=len(ranked_pool), target_n=n_per_beam,
            matchmaker_diag=beam_mm_diag,
        )

        if beam_id == "Z" and two_stage_z:
            # =================================================================
            # Two-stage Z: Phase 1 (pormake+lammps+zeo) → filter → Phase 2 (RASPA)
            # =================================================================
            print(f"[Beam Z] Phase 1: pormake+lammps+zeo for {len(ranked_pool)} candidates")
            stage1_results: List[Tuple] = []

            for ranked_mof in ranked_pool:
                comp = ranked_mof.component
                cached = sim_cache.get(comp.topology, comp.node, comp.edge)
                if cached is not None and cached.status == "success":
                    # F10: re-check cached result against current geometry_filter
                    cached_geom = cached.real_geometry or cached.predicted_geometry
                    if target_geometry and cached_geom and not passes_geometry_filter(cached_geom, target_geometry):
                        print(f"   [CACHE STALE] {comp.filename} → fails current geometry_filter, re-sim")
                    else:
                        print(f"   [CACHE HIT] {comp.filename} → success")
                        stage1_results.append((ranked_mof, cached))
                        continue

                mof_work_dir = os.path.join(iter_dir, comp.filename)
                os.makedirs(mof_work_dir, exist_ok=True)
                r = _simulate_one_mof(
                    comp, mof_work_dir,
                    predicted_geom=ranked_mof.predicted_geometry,
                    match_score=ranked_mof.match_score,
                    use_zeo=True, zeopp_bin=zeopp_bin,
                    pipeline="stage1_only",
                )
                stage1_results.append((ranked_mof, r))

            # Filter by geometry_filter
            all_stage1_done = [
                (rm, r) for rm, r in stage1_results
                if r.status in ("stage1_done", "success") and r.real_geometry
            ]
            passing = [
                (rm, r) for rm, r in all_stage1_done
                if passes_geometry_filter(r.real_geometry, target_geometry)
            ]
            n_strict = len(passing)
            print(f"[Beam Z] Phase 1: {n_strict}/{len(stage1_results)} pass strict geometry filter")

            if n_strict == 0:
                # Compute actual geometry distribution for Agent 1 feedback
                geo_keys = [
                    ("di",      "target_Di_min",      "target_Di_max"),
                    ("df",      "target_Df_min",      "target_Df_max"),
                    ("sa",      "target_sa_min",      "target_sa_max"),
                    ("vf",      "target_vf_min",      "target_vf_max"),
                    ("density", "target_density_min", "target_density_max"),
                ]
                stats: Dict[str, Any] = {}
                for gkey, min_key, max_key in geo_keys:
                    vals = [r.real_geometry[gkey] for _, r in all_stage1_done
                            if r.real_geometry and gkey in r.real_geometry]
                    if vals:
                        stats[gkey] = {
                            "mean": round(sum(vals) / len(vals), 3),
                            "min": round(min(vals), 3),
                            "max": round(max(vals), 3),
                            "target_min": target_geometry.get(min_key),
                            "target_max": target_geometry.get(max_key),
                        }
                live_results.geometry_aborted = True
                live_results.stage1_geometry_stats = stats
                print(f"[Beam Z] GEOMETRY ABORT: 0/{len(all_stage1_done)} pass geometry filter "
                      f"→ aborting iteration (skipping A/F/total beams)")
                break  # Skip A/F/total — caller will generate geometry redesign feedback

            # Conditional fallback: if strict passes < target, fill with best non-passing by geometry score
            n_top = config.LIVE_SIM_Z_RASPA_TOP
            if n_strict < n_top:
                not_passing = [
                    (rm, r) for rm, r in all_stage1_done
                    if not passes_geometry_filter(r.real_geometry, target_geometry)
                ]
                not_passing.sort(
                    key=lambda x: compute_geometry_match_score(x[1].real_geometry, target_geometry),
                    reverse=True,
                )
                n_fallback = min(n_top - n_strict, len(not_passing))
                z_top = passing[:n_top] + not_passing[:n_fallback]
                # Pre-compute fail reasons for fallback candidates
                for _, r in not_passing[:n_fallback]:
                    r.geo_filter_fail_reason = geometry_fail_reason(r.real_geometry, target_geometry)
                print(f"[Beam Z] Phase 2: {n_strict} strict + {n_fallback} fallback → {len(z_top)} RASPA candidates")
            else:
                z_top = passing[:n_top]
                print(f"[Beam Z] Phase 2: {n_strict} strict → {len(z_top)} RASPA candidates (no fallback needed)")

            strict_filenames = {r.filename for _, r in passing[:n_top]}

            for ranked_mof, s1_result in z_top:
                if len(beam_result.successes) >= n_per_beam:
                    break

                comp = ranked_mof.component
                is_strict = comp.filename in strict_filenames

                if s1_result.status == "success":
                    # Cache hit — already has RASPA uptake
                    s1_result.geo_filter_passed = is_strict
                    s1_result.geo_filter_fail_reason = "" if is_strict else s1_result.geo_filter_fail_reason
                    beam_result.successes.append(s1_result)
                    live_results.n_real_simulations += 1
                    continue

                mof_work_dir = os.path.join(iter_dir, comp.filename)
                os.makedirs(mof_work_dir, exist_ok=True)
                result = _simulate_one_mof(
                    comp, mof_work_dir,
                    predicted_geom=ranked_mof.predicted_geometry,
                    match_score=ranked_mof.match_score,
                    pipeline="stage2_only",
                    stage2_cif_path=s1_result.cif_path,
                    stage2_real_geometry=s1_result.real_geometry,
                )
                result.geo_filter_passed = is_strict
                result.geo_filter_fail_reason = "" if is_strict else s1_result.geo_filter_fail_reason
                sim_cache.put(result)

                if result.status == "success":
                    beam_result.successes.append(result)
                    live_results.n_real_simulations += 1
                    print(f"   [OK] {comp.filename}: "
                          f"{_ads.upper()}={result.real_uptake.get('xe_loading_mol_kg' if _ads == 'xekr' else 'loading_mol_kg', '?')} mol/kg "
                          f"({result.wall_seconds:.0f}s)")
                else:
                    beam_result.failures.append(result)
                    live_results.n_failures += 1
                    print(f"   [FAIL] {comp.filename}: {result.status} — "
                          f"{result.error_msg[:100]} ({result.wall_seconds:.0f}s)")

            # Add stage1 build failures to beam failures
            for _, r in stage1_results:
                if r.status not in ("stage1_done", "success"):
                    beam_result.failures.append(r)
                    live_results.n_failures += 1

        elif use_zeo:
            # =================================================================
            # Two-stage A/F/total: Phase 1 (pormake+lammps+zeo) → Phase 2 (RASPA)
            # =================================================================
            print(f"[Beam {beam_id}] Phase 1: pormake+lammps+zeo for {len(ranked_pool)} candidates")
            af_stage1_results: List[Tuple] = []

            for ranked_mof in ranked_pool:
                comp = ranked_mof.component
                cached = sim_cache.get(comp.topology, comp.node, comp.edge)
                if cached is not None and cached.status in ("stage1_done", "success"):
                    print(f"   [CACHE HIT] {comp.filename} → {cached.status}")
                    af_stage1_results.append((ranked_mof, cached))
                    continue

                mof_work_dir = os.path.join(iter_dir, comp.filename)
                os.makedirs(mof_work_dir, exist_ok=True)
                r = _simulate_one_mof(
                    comp, mof_work_dir,
                    predicted_geom=ranked_mof.predicted_geometry,
                    match_score=ranked_mof.match_score,
                    use_zeo=True, zeopp_bin=zeopp_bin,
                    pipeline="stage1_only",
                )
                af_stage1_results.append((ranked_mof, r))

            af_stage1_done = [
                (rm, r) for rm, r in af_stage1_results
                if r.status in ("stage1_done", "success") and r.cif_path
            ]
            af_top = af_stage1_done[:n_per_beam]
            print(f"[Beam {beam_id}] Phase 1: {len(af_stage1_done)}/{len(af_stage1_results)} success "
                  f"→ {len(af_top)} RASPA candidates")

            for ranked_mof, s1_result in af_top:
                comp = ranked_mof.component

                if s1_result.status == "success":
                    # Already has RASPA uptake (cache hit)
                    beam_result.successes.append(s1_result)
                    live_results.n_real_simulations += 1
                    continue

                mof_work_dir = os.path.join(iter_dir, comp.filename)
                os.makedirs(mof_work_dir, exist_ok=True)
                result = _simulate_one_mof(
                    comp, mof_work_dir,
                    predicted_geom=ranked_mof.predicted_geometry,
                    match_score=ranked_mof.match_score,
                    pipeline="stage2_only",
                    stage2_cif_path=s1_result.cif_path,
                    stage2_real_geometry=s1_result.real_geometry,
                )
                sim_cache.put(result)

                if result.status == "success":
                    beam_result.successes.append(result)
                    live_results.n_real_simulations += 1
                    print(f"   [OK] {comp.filename}: "
                          f"{_ads.upper()}={result.real_uptake.get('xe_loading_mol_kg' if _ads == 'xekr' else 'loading_mol_kg', '?')} mol/kg "
                          f"({result.wall_seconds:.0f}s)")
                else:
                    beam_result.failures.append(result)
                    live_results.n_failures += 1
                    print(f"   [FAIL] {comp.filename}: {result.status} — "
                          f"{result.error_msg[:100]} ({result.wall_seconds:.0f}s)")

            # Stage1 failures
            for _, r in af_stage1_results:
                if r.status not in ("stage1_done", "success"):
                    beam_result.failures.append(r)
                    live_results.n_failures += 1

        else:
            # =================================================================
            # Standard pipeline (--no-zeo): full simulation per candidate
            # =================================================================
            for ranked_mof in ranked_pool:
                if len(beam_result.successes) >= n_per_beam:
                    break

                comp = ranked_mof.component

                cached = sim_cache.get(comp.topology, comp.node, comp.edge)
                if cached is not None:
                    if cached.status != "success":
                        print(f"   [CACHE HIT] {comp.filename} → {cached.status} (skip)")
                        beam_result.failures.append(cached)
                        continue
                    print(f"   [CACHE HIT] {comp.filename} → success")
                    beam_result.successes.append(cached)
                    continue

                mof_work_dir = os.path.join(iter_dir, comp.filename)
                os.makedirs(mof_work_dir, exist_ok=True)

                result = _simulate_one_mof(
                    comp, mof_work_dir,
                    predicted_geom=ranked_mof.predicted_geometry,
                    match_score=ranked_mof.match_score,
                    use_zeo=False,
                )
                sim_cache.put(result)

                if result.status == "success":
                    beam_result.successes.append(result)
                    live_results.n_real_simulations += 1
                    print(f"   [OK] {comp.filename}: "
                          f"{_ads.upper()}={result.real_uptake.get('xe_loading_mol_kg' if _ads == 'xekr' else 'loading_mol_kg', '?')} mol/kg "
                          f"({result.wall_seconds:.0f}s)")
                else:
                    beam_result.failures.append(result)
                    live_results.n_failures += 1
                    print(f"   [FAIL] {comp.filename}: {result.status} — "
                          f"{result.error_msg[:100]} ({result.wall_seconds:.0f}s)")

        # Report beam status
        n_ok = len(beam_result.successes)
        n_fail = len(beam_result.failures)
        if beam_result.is_complete:
            print(f"[Beam {beam_id}] COMPLETE: {n_ok}/{n_per_beam} successes "
                  f"({n_fail} failures)")
        elif beam_result.is_acceptable:
            print(f"[Beam {beam_id}] PARTIAL: {n_ok}/{n_per_beam} successes "
                  f"(acceptable, {n_fail} failures)")
        else:
            print(f"[Beam {beam_id}] ABORTED: only {n_ok}/{n_per_beam} successes "
                  f"({n_fail} failures, pool exhausted)")
            live_results.aborted_beams.append(beam_id)

        live_results.beams[beam_id] = beam_result

    live_results.wall_clock_seconds = time.time() - t_start

    # Save iteration summary
    summary_path = os.path.join(run_dir, f"iter_{iteration}", "live_results.json")
    _save_iteration_summary(live_results, summary_path)

    print(f"\n{'='*60}")
    print(f"[Live Runner] Iteration {iteration} complete in "
          f"{live_results.wall_clock_seconds:.0f}s")
    print(f"  Simulations: {live_results.n_real_simulations} real, "
          f"{live_results.n_failures} failures")
    print(f"  Cache size: {len(sim_cache)} entries")
    if live_results.aborted_beams:
        print(f"  ABORTED beams: {live_results.aborted_beams}")
    print(f"{'='*60}")

    return live_results


def prepare_beam_pools(
    specs: Dict[str, Any],
    matchmaker,
    n_per_beam: int = config.LIVE_SIM_N_PER_BEAM,
    z_pool_size: Optional[int] = None,
    af_pool_size: Optional[int] = None,
) -> Dict[str, List[Any]]:
    """
    Run matchmaker + mof2zeo ranking for all 4 beams.
    Returns {beam_id: [RankedMOF, ...]} — the ranked candidate pools
    ready for manifest generation (no simulation).

    Args:
        z_pool_size: Z beam pool size (default: n_per_beam * LIVE_SIM_POOL_MULTIPLIER).
            Typically LIVE_SIM_Z_POOL_SIZE=100 for two-stage mode.
        af_pool_size: A/F/total beam pool size (default: n_per_beam * LIVE_SIM_POOL_MULTIPLIER).
            Typically LIVE_SIM_AF_POOL_SIZE=30 for two-stage mode.
    """
    predictor = GeometryPredictor()
    ranker = MOFRanker()
    gen = ComponentGenerator()

    target_geometry = specs.get("geometry_filter", {})

    beam_configs = [
        ("Z", "Full Hypothesis (Chemistry + Geometry)", config.LIVE_SIM_POOL_MULTIPLIER),
        ("A", "Chemistry Only (no geometry gate)", config.LIVE_SIM_POOL_MULTIPLIER),
        ("F", "Metal Only (no linker constraints)", config.LIVE_SIM_POOL_MULTIPLIER),
        ("total", "Random Baseline (global)", config.LIVE_SIM_POOL_MULTIPLIER_RANDOM),
    ]

    beam_pools: Dict[str, List[Any]] = {}

    for beam_id, beam_label, pool_mult in beam_configs:
        print(f"\n[Prepare] Beam {beam_id}: {beam_label}")
        two_stage_z = bool(z_pool_size) and _has_geometry_constraints(target_geometry)
        if beam_id == "Z" and two_stage_z:
            pool_size = z_pool_size
        elif beam_id != "Z" and af_pool_size is not None:
            pool_size = af_pool_size
        elif beam_id == "Z" and af_pool_size is not None:
            pool_size = af_pool_size
        else:
            pool_size = n_per_beam * pool_mult

        beam_preferred = {}
        if beam_id == "total":
            _cv_thr = config.LIVE_SIM_CV_THRESHOLD
            _raw_size = 1000 if _cv_thr is not None else pool_size
            components = _build_random_pool(_raw_size)
            print(f"[Prepare] Random pool: {len(components)} candidates")
        else:
            beam_specs = _build_beam_specs(specs, beam_id)
            mm_result = matchmaker.smart_matchmaker_single_node(beam_specs)
            beam_preferred = mm_result.get("preferred_features", {})

            if mm_result.get("status") == "error":
                print(f"[Prepare] Beam {beam_id} matchmaker error: {mm_result.get('message')}")
                beam_pools[beam_id] = []
                continue

            mm_result = filter_matchmaker_result(mm_result)

            if not mm_result.get("topology"):
                print(f"[Prepare] Beam {beam_id} no simulation-safe topologies")
                beam_pools[beam_id] = []
                continue

            try:
                components = gen.generate_from_matchmaker(mm_result, max_combos=config.LIVE_SIM_MAX_COMBOS)
            except ValueError as e:
                print(f"[Prepare] Beam {beam_id} ComponentGenerator error: {e}")
                beam_pools[beam_id] = []
                continue

        cv_threshold = config.LIVE_SIM_CV_THRESHOLD
        if beam_id == "Z" and _has_geometry_constraints(target_geometry):
            print(f"[Prepare] Predicting geometry for {len(components)} candidates (mof2zeo)...")
            predictions = predictor.predict_batch(components)
            if cv_threshold is not None:
                components, predictions = _apply_cv_filter(components, predictions, cv_threshold, beam_id)
            ranked = ranker.rank(
                components, predictions, _expand_geometry_filter(target_geometry),
                preferred_features=beam_preferred or None,
                bb_lookup=matchmaker.bb_lookup if beam_preferred else None,
            ) if components else []
        else:
            if cv_threshold is not None:
                print(f"[Prepare] Beam {beam_id}: predicting geometry for CV filter ({len(components)} candidates)...")
                predictions = predictor.predict_batch(components)
                components, _ = _apply_cv_filter(components, predictions, cv_threshold, beam_id)
            else:
                print(f"[Prepare] Beam {beam_id}: no geometry filter → skipping mof2zeo (random pool)")
            ranked = _random_ranked_pool(components)

        # Metal-stratified selection when enabled (rare-but-best metals reach the sim pool);
        # firewall: identity only; fail-safe -> top-N. Gated by config.STRATIFIED_SAMPLING.
        if config.is_stratified_sampling():
            from core import sampling_strata
            ranked_pool = sampling_strata.stratified_rank_select(ranked, pool_size)
        else:
            ranked_pool = ranked[:pool_size]
        if ranked_pool:
            print(f"[Prepare] Beam {beam_id}: {len(ranked_pool)} candidates "
                  f"(top score={ranked_pool[0].match_score:.3f})")
        else:
            print(f"[Prepare] Beam {beam_id}: empty pool")

        beam_pools[beam_id] = ranked_pool

    total_jobs = sum(len(pool) for pool in beam_pools.values())
    print(f"\n[Prepare] Total: {total_jobs} candidates across {len(beam_pools)} beams")
    return beam_pools


def _save_iteration_summary(results: LiveResults, path: str) -> None:
    """Save a JSON summary of the iteration results."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary = {
        "n_real_simulations": results.n_real_simulations,
        "n_failures": results.n_failures,
        "wall_clock_seconds": results.wall_clock_seconds,
        "aborted_beams": results.aborted_beams,
        "beams": {},
    }
    for beam_id, beam in results.beams.items():
        summary["beams"][beam_id] = {
            "beam_label": beam.beam_label,
            "n_successes": len(beam.successes),
            "n_failures": len(beam.failures),
            "pool_size": beam.pool_size,
            "target_n": beam.target_n,
            "successes": [s.to_dict() for s in beam.successes],
            "failures": [f.to_dict() for f in beam.failures],
        }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Live Runner] Saved iteration summary to {path}")
