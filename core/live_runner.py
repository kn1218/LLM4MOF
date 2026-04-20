"""
Live Simulation Runner — generator-style MOF simulation with refill-on-failure.

Orchestrates the full pipeline per beam:
  matchmaker → HAN_SAFE filter → mof2zeo prefilter → ranked pool
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
from core.han_safe_topologies import HAN_SAFE_TOPOS, filter_matchmaker_result
from core.filter_candidate import (
    MOFComponent, PredictedGeometry, GeometryPredictor, MOFRanker,
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
    status: str          # "success" | "build_fail" | "lammps_fail" | "raspa_fail"
    predicted_geometry: Optional[Dict[str, float]] = None
    match_score: float = 0.0
    real_uptake: Optional[Dict[str, float]] = None   # parsed RASPA output
    error_msg: str = ""
    wall_seconds: float = 0.0

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
) -> SimResult:
    """
    Run the full PORMAKE → LAMMPS → RASPA3 pipeline for one MOF.

    Returns a SimResult with status and parsed uptake (if successful).
    """
    t0 = time.time()
    filename = component.filename
    pred_dict = predicted_geom.to_dict() if predicted_geom else None

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
    opt_cif_dir = cif_dir  # default: use un-optimized CIF
    if not config.LIVE_SIM_SKIP_LAMMPS:
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
                print(f"   [LAMMPS] lammps-interface failed; using un-optimized CIF")
            else:
                # Step 2: Create optimization input + run LAMMPS
                created = make_optimization_input(data_dir, lammps_out)
                if created:
                    opt_ok = run_lammps_optimization(created[0], lammps_out)
                    if opt_ok:
                        # Step 3: Convert optimized data → CIF
                        opt_data_dir = lammps_out / "opt_lammps_data"
                        opt_cif_out = lammps_out / "optimized_cifs"
                        opt_cif_out.mkdir(exist_ok=True, parents=True)
                        lammps_files = list(opt_data_dir.glob("*_opt.lammps-data"))
                        if lammps_files:
                            name = lammps_files[0].stem.replace("data.", "").replace("_opt", "")
                            convert_lammps_to_cif(lammps_files[0], opt_cif_out / f"{name}.cif")
                            if any(f.name.endswith(".cif") for f in opt_cif_out.iterdir()):
                                opt_cif_dir = str(opt_cif_out)
                                print(f"   [LAMMPS] Using optimized CIF")
                            else:
                                print(f"   [LAMMPS] Conversion failed; using original")
                        else:
                            print(f"   [LAMMPS] No optimized data produced; using original")
                    else:
                        print(f"   [LAMMPS] Optimization failed; using un-optimized CIF")
                else:
                    print(f"   [LAMMPS] No input files created; using un-optimized CIF")
        except Exception as e:
            print(f"   [LAMMPS] Error ({e}); using un-optimized CIF")

    # --- Stage 3: RASPA3 GCMC ---
    try:
        from core.simulation.gcmc.run_raspa import (
            create_raspa_input, run_simulation_background, parse_output,
            DEFAULT_FORCEFIELD_DIR, DEFAULT_RASPA3,
        )

        raspa_out = os.path.join(work_dir, "raspa_output")
        mof_dict = {"filename": filename}
        params = {
            "cycles": config.LIVE_SIM_RASPA_CYCLES,
            "init_cycles": config.LIVE_SIM_RASPA_INIT_CYCLES,
            "temperature": config.LIVE_SIM_RASPA_TEMPERATURE,
            "pressure": config.LIVE_SIM_RASPA_PRESSURE,
            "forcefield_dir": DEFAULT_FORCEFIELD_DIR,
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
            DEFAULT_FORCEFIELD_DIR, filename, raspa3_bin,
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
        parsed = parse_output(mof_output_dir)

        if not parsed or not parsed.get("loading_mol_kg"):
            return SimResult(
                topology=component.topology, node=component.node,
                edge=component.edge, filename=filename,
                status="raspa_fail", predicted_geometry=pred_dict,
                match_score=match_score,
                error_msg="RASPA3 output parsing failed (no loading data)",
                wall_seconds=time.time() - t0,
            )

        return SimResult(
            topology=component.topology, node=component.node,
            edge=component.edge, filename=filename,
            status="success", predicted_geometry=pred_dict,
            match_score=match_score, real_uptake=parsed,
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
    Build a random pool from HAN_SAFE_TOPOS × all nodes × all edges.
    Used for Beam 4 (global baseline).
    """
    from core.filter_candidate import ComponentGenerator

    gen = ComponentGenerator()
    safe_topos = list(HAN_SAFE_TOPOS)

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

    for beam_id, beam_label, pool_mult in beam_configs:
        print(f"\n{'='*60}")
        print(f"[Beam {beam_id}] {beam_label}")
        print(f"{'='*60}")

        pool_size = n_per_beam * pool_mult
        iter_dir = os.path.join(run_dir, f"iter_{iteration}", f"beam_{beam_id}")
        os.makedirs(iter_dir, exist_ok=True)

        # --- Build ranked candidate pool ---
        beam_mm_diag = {}
        beam_preferred = {}  # preferred_features for soft ranking bonus
        if beam_id == "total":
            # Random baseline: skip matchmaker, sample from full space
            components = _build_random_pool(pool_size)
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

            # Apply HAN_SAFE filter
            mm_result = filter_matchmaker_result(mm_result)

            if not mm_result.get("topology"):
                print(f"[Beam {beam_id}] No HAN-safe topologies after filter")
                live_results.beams[beam_id] = BeamResult(
                    beam_id=beam_id, beam_label=beam_label,
                    pool_size=0, target_n=n_per_beam,
                    matchmaker_diag=beam_mm_diag,
                )
                live_results.aborted_beams.append(beam_id)
                continue

            # Generate combinations
            try:
                components = gen.generate_from_matchmaker(mm_result)
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
        print(f"[Beam {beam_id}] Predicting geometry for {len(components)} candidates...")
        predictions = predictor.predict_batch(components)
        ranked = ranker.rank(
            components, predictions, target_geometry,
            preferred_features=beam_preferred or None,
            bb_lookup=matchmaker.bb_lookup if beam_preferred else None,
        )

        # Take top pool_size candidates
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

        # --- Generator-style simulation with refill ---
        beam_result = BeamResult(
            beam_id=beam_id, beam_label=beam_label,
            pool_size=len(ranked_pool), target_n=n_per_beam,
            matchmaker_diag=beam_mm_diag,
        )

        for ranked_mof in ranked_pool:
            if len(beam_result.successes) >= n_per_beam:
                break

            comp = ranked_mof.component

            # Check cache first
            cached = sim_cache.get(comp.topology, comp.node, comp.edge)
            if cached is not None:
                if cached.status == "success":
                    print(f"   [CACHE HIT] {comp.filename} → success")
                    beam_result.successes.append(cached)
                else:
                    print(f"   [CACHE HIT] {comp.filename} → {cached.status} (skip)")
                    beam_result.failures.append(cached)
                continue

            # Run simulation
            print(f"   [SIM {len(beam_result.successes)+1}/{n_per_beam}] "
                  f"{comp.filename} (score={ranked_mof.match_score:.3f})")

            mof_work_dir = os.path.join(iter_dir, comp.filename)
            os.makedirs(mof_work_dir, exist_ok=True)

            result = _simulate_one_mof(
                comp, mof_work_dir,
                predicted_geom=ranked_mof.predicted_geometry,
                match_score=ranked_mof.match_score,
            )

            # Persist to cache
            sim_cache.put(result)

            if result.status == "success":
                beam_result.successes.append(result)
                live_results.n_real_simulations += 1
                print(f"   [OK] {comp.filename}: "
                      f"H2={result.real_uptake.get('loading_mol_kg', '?')} mol/kg "
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
) -> Dict[str, List[Any]]:
    """
    Run matchmaker + mof2zeo ranking for all 4 beams.
    Returns {beam_id: [RankedMOF, ...]} — the ranked candidate pools
    ready for manifest generation (no simulation).
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
        pool_size = n_per_beam * pool_mult

        beam_preferred = {}
        if beam_id == "total":
            components = _build_random_pool(pool_size)
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
                print(f"[Prepare] Beam {beam_id} no HAN-safe topologies")
                beam_pools[beam_id] = []
                continue

            try:
                components = gen.generate_from_matchmaker(mm_result)
            except ValueError as e:
                print(f"[Prepare] Beam {beam_id} ComponentGenerator error: {e}")
                beam_pools[beam_id] = []
                continue

        print(f"[Prepare] Predicting geometry for {len(components)} candidates...")
        predictions = predictor.predict_batch(components)
        ranked = ranker.rank(
            components, predictions, target_geometry,
            preferred_features=beam_preferred or None,
            bb_lookup=matchmaker.bb_lookup if beam_preferred else None,
        )

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
