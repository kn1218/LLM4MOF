#!/usr/bin/env python3
"""
Single-MOF simulation script for HPC array jobs.

Reads a job from batch_manifest.json by index, runs:
  PORMAKE build → LAMMPS optimize → RASPA3 GCMC
Writes a per-MOF result JSON.

Usage:
  python run_mof_sim.py --manifest batch_manifest.json --job-index 0 --output-dir results/
  python run_mof_sim.py --manifest batch_manifest.json --job-index $PBS_ARRAYID --output-dir results/
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

sys.setrecursionlimit(10000)
import time
from pathlib import Path

# --- project root on sys.path for config / raspa_utils imports ---
_hpc_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_hpc_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

try:
    from config import LIVE_SIM_ADSORBATE_CONFIGS
except ImportError:
    LIVE_SIM_ADSORBATE_CONFIGS = None

try:
    from core.simulation.gcmc.raspa_utils import parse_output_mixture as _parse_mixture
except ImportError:
    _parse_mixture = None

# ── Inline XeKr parser (fallback when raspa_utils is not importable on HPC) ──
_FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_COMPONENT_HEADER_RE = re.compile(r"Component\s+\d+\s+[\[\(](.+?)[\]\)]")
_ABS_STEP_MOLKG_RE = re.compile(
    rf"absolute adsorption:.*?({_FLOAT_RE})(?:\s+\(avg\.\s+{_FLOAT_RE}\))?\s+\[mol/uc\],\s*"
    rf"({_FLOAT_RE})(?:\s+\(avg\.\s+{_FLOAT_RE}\))?\s+\[mol/kg\]"
)
_ABS_LOADING_AVG_MOLKG_RE = re.compile(
    rf"Abs\.\s+loading\s+average\s+({_FLOAT_RE}).*\[mol/kg"
)


def _parse_mixture_inline(raspa_dir: str, xe_molfrac: float = 0.20,
                          cif_path: str = None) -> dict:
    """Inline XeKr mixture parser — used when raspa_utils cannot be imported."""
    output_subdir = os.path.join(raspa_dir, "output")
    search_dir = output_subdir if os.path.isdir(output_subdir) else raspa_dir

    # Find newest .txt output file
    candidates = [
        os.path.join(search_dir, f) for f in os.listdir(search_dir)
        if f.startswith("output_") and f.endswith(".txt")
    ] if os.path.isdir(search_dir) else []
    if not candidates:
        return None
    output_file = max(candidates, key=os.path.getmtime)

    with open(output_file) as f:
        lines = f.readlines()

    kr_molfrac = 1.0 - xe_molfrac
    loadings = {}
    current_component = None

    for line in lines:
        m = _COMPONENT_HEADER_RE.search(line)
        if m:
            current_component = m.group(1).strip()
            continue
        if current_component is not None:
            m = _ABS_STEP_MOLKG_RE.search(line)
            if m:
                loadings[current_component] = float(m.group(2))
            else:
                m = _ABS_LOADING_AVG_MOLKG_RE.search(line)
                if m:
                    loadings[current_component] = float(m.group(1))

    xe_mol_kg = loadings.get("Xe")
    kr_mol_kg = loadings.get("Kr")
    if xe_mol_kg is None or kr_mol_kg is None:
        return None

    result = {"xe_loading_mol_kg": xe_mol_kg, "kr_loading_mol_kg": kr_mol_kg}
    if kr_mol_kg > 0 and kr_molfrac > 0 and xe_molfrac > 0:
        result["selectivity_xe_kr"] = (xe_mol_kg / kr_mol_kg) / (xe_molfrac / kr_molfrac)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run single MOF simulation")
    parser.add_argument("--manifest", required=True, help="Path to batch_manifest.json")
    parser.add_argument("--job-index", type=int, required=True, help="Job index (0-based)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--forcefield-dir", default=None, help="Path to forcefield dir; auto-selected from --adsorbate if not set")
    parser.add_argument("--adsorbate", default="h2", choices=["h2", "ch4", "co2", "xekr"],
                        help="Adsorbate type (default: h2)")
    parser.add_argument("--xe-molfrac", type=float, default=0.20,
                        help="Xe mole fraction for xekr (default: 0.20)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override RASPA temperature in K (default: adsorbate config value)")
    parser.add_argument("--pressure", type=float, default=None,
                        help="Override RASPA pressure in bar (default: adsorbate config value)")
    parser.add_argument("--use-zeo", action="store_true", help="Run Zeo++ after RASPA3 to compute actual geometry")
    parser.add_argument("--zeopp-bin", default=None, help="Path to Zeo++ network binary")
    return parser.parse_args()


def load_job(manifest_path: str, job_index: int) -> tuple:
    """Load manifest and extract job by index."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    jobs = manifest["jobs"]
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"Job index {job_index} out of range (0-{len(jobs)-1})")
    return manifest["config"], jobs[job_index]


def stage_pormake(job: dict, work_dir: str) -> tuple:
    """Build MOF CIF via PORMAKE. Returns (success, cif_path, error_msg, seconds)."""
    t0 = time.time()
    cif_dir = os.path.join(work_dir, "cif")
    os.makedirs(cif_dir, exist_ok=True)

    try:
        import pormake as pm
        database = pm.Database()
        builder = pm.Builder()

        topology = database.get_topo(job["topology"])
        node_bb = database.get_bb(job["node"])
        edge_bb = database.get_bb(job["edge"])

        topology.describe()

        if topology.n_node_types > 1:
            return False, None, f"Multi-node-type topology ({topology.n_node_types})", time.time() - t0

        node_bbs = {0: node_bb}
        cn = topology.cn[0]
        if node_bb.n_connection_points != cn:
            return False, None, f"CN mismatch: node={node_bb.n_connection_points}, topo={cn}", time.time() - t0

        edge_bbs = {(0, 0): edge_bb}
        mof = builder.build_by_type(topology, node_bbs, edge_bbs)

        filename = job["filename"]
        cif_path = os.path.join(cif_dir, f"{filename}.cif")
        mof.write_cif(cif_path)
        print(f"   [PORMAKE] OK: {filename}.cif")
        return True, cif_path, "", time.time() - t0

    except Exception as e:
        return False, None, str(e)[:300], time.time() - t0


def stage_lammps(cif_path: str, work_dir: str, cfg: dict) -> tuple:
    """LAMMPS optimization. Returns (success, active_cif_path, error_msg, seconds).

    Uses lammps_interface to generate LAMMPS data+input files from CIF,
    runs LAMMPS minimization, writes optimized data file, and converts back
    to CIF. Returns the optimized CIF path on success; falls back to the
    original cif_path on non-fatal errors.
    """
    if cfg.get("skip_lammps", False):
        return True, cif_path, "skipped", 0.0

    t0 = time.time()
    timeout = cfg.get("lammps_timeout", 900)
    stem = Path(cif_path).stem

    try:
        from lammps_interface.lammps_main import LammpsSimulation
        from lammps_interface.structure_data import from_CIF
        from lammps_interface.InputHandler import Options
        from types import SimpleNamespace

        lammps_dir = os.path.join(work_dir, "lammps")
        os.makedirs(lammps_dir, exist_ok=True)

        # Build options namespace — LammpsSimulation expects options.cif_file
        # and from_CIF parses the CIF into (cell, graph) separately
        options = SimpleNamespace(
            cif_file=cif_path,
            force_field="UFF",
            minimize=False,
            orthogonalize=False,
            replication="1x1x1",
            cutoff=12.8,
            dreid_bond_type="harmonic",
            h_bonding=False,
            fix_metal=False,
            mol_ff=None,
            neighbour_size=2.0,
            random_vel=False,
            temp=298.15,
            pressure=100.0,
            npt=False,
            nvt=False,
            bulk_moduli=False,
            thermal_scaling=False,
            max_dev=0.01,
            tol=0.0001,
            iter_count=500,
            neqstp=5000,
            nprodstp=5000,
            dump_dcd=False,
            dump_xyz=False,
            dump_lammpstrj=False,
            restart=False,
            output_cif=False,
            output_pdb=False,
            output_raspa=False,
        )

        sim = LammpsSimulation(options)
        cell, graph = from_CIF(cif_path)
        sim.set_cell(cell)
        sim.set_graph(graph)
        sim.split_graph()
        sim.assign_force_fields()
        sim.compute_simulation_size()
        sim.merge_graphs()
        sim.write_lammps_files(lammps_dir)
        print(f"   [LAMMPS] lammps-interface OK for {stem}")

        # Find generated input file
        in_files = list(Path(lammps_dir).glob("in.*"))
        if not in_files:
            return False, None, "lammps-interface no input file", time.time() - t0

        input_file = str(in_files[0])

        # Append aniso alternating minimize block (atomic → cell+atomic × 2 loops,
        # then final atomic). aniso allows each axis to relax independently, which is
        # correct for non-cubic MOFs. iso (the previous default) forced uniform scaling.
        opt_data_path = os.path.join(lammps_dir, f"data.{stem}_opt.lammps-data")
        minimize_block = (
            "\nthermo_style custom step pe press pxx pyy pzz lx ly lz xy xz yz\n"
            "thermo 1000\n"
            "min_style cg\n"
            "\nvariable llm4mof_i loop 2\n"
            "label llm4mof_loop\n"
            "\nminimize 1.0e-4 1.0e-6 1000 10000\n"
            "\nfix llm4mof_relax all box/relax aniso 0.0 vmax 0.001\n"
            "minimize 1.0e-4 1.0e-6 1000 10000\n"
            "unfix llm4mof_relax\n"
            "\nnext llm4mof_i\n"
            "jump SELF llm4mof_loop\n"
            "\nminimize 1.0e-4 1.0e-6 1000 10000\n"
            "\nrun 0\n"
            f"\nwrite_data  {opt_data_path}\n"
        )
        with open(input_file, "a") as f:
            f.write(minimize_block)

        # Run LAMMPS
        lmp_bin = shutil.which("lmp_mpi") or shutil.which("lmp")
        if not lmp_bin:
            return False, None, "lmp not found", time.time() - t0

        result = subprocess.run(
            [lmp_bin, "-in", input_file],
            cwd=lammps_dir,
            capture_output=True, text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return False, None, f"lammps exit {result.returncode}", time.time() - t0

        print(f"   [LAMMPS] Optimization OK for {stem}")

        # Convert optimized LAMMPS data → CIF
        if not os.path.isfile(opt_data_path):
            return False, None, "write_data output missing", time.time() - t0

        try:
            # Inline ASE-based LAMMPS data → CIF conversion (no core/ dependency on HPC)
            from ase.io import read as ase_read
            from ase.data import atomic_masses, chemical_symbols
            opt_cif_dir = os.path.join(lammps_dir, "optimized_cifs")
            os.makedirs(opt_cif_dir, exist_ok=True)
            opt_cif_path = os.path.join(opt_cif_dir, f"{stem}.cif")
            atoms = ase_read(opt_data_path, format="lammps-data", style="full")
            for atom in atoms:
                min_val = min(atomic_masses, key=lambda m: abs(m - atom.mass))
                atom.symbol = chemical_symbols[atomic_masses.tolist().index(min_val)]
            atoms.write(opt_cif_path)
            if os.path.isfile(opt_cif_path):
                print(f"   [LAMMPS] Using optimized CIF for {stem}")
                return True, opt_cif_path, "", time.time() - t0
            else:
                return False, None, "CIF conversion failed", time.time() - t0
        except Exception as e:
            return False, None, f"CIF conversion error: {str(e)[:200]}", time.time() - t0

    except subprocess.TimeoutExpired:
        return False, None, f"lammps timeout ({timeout}s)", time.time() - t0
    except Exception as e:
        return False, None, str(e)[:200], time.time() - t0


def _assign_framework_charges(cif_path: str) -> None:
    """Assign DDEC6 partial atomic charges using PACMAN-charge (in-place, CO2 only)."""
    try:
        from PACMANCharge import pmcharge
    except ImportError:
        print("   [CHARGE] PACMAN-charge not installed; skipping charge assignment")
        print("   [CHARGE] To install: pip install PACMAN-charge")
        return
    pacman_cif = cif_path.replace(".cif", "_pacman.cif")
    try:
        pmcharge.predict(cif_file=cif_path, charge_type="DDEC6", digits=6, atom_type=True, neutral=True, keep_connect=True)
        if os.path.isfile(pacman_cif):
            os.replace(pacman_cif, cif_path)
            print(f"   [CHARGE] DDEC6 charges assigned: {os.path.basename(cif_path)}")
        else:
            print(f"   [CHARGE] PACMAN output not found; using original CIF")
    except Exception as e:
        print(f"   [CHARGE] PACMAN-charge failed ({e}); using original CIF")
        if os.path.isfile(pacman_cif):
            os.remove(pacman_cif)


def get_unit_cells(cif_path: str, cutoff: float = 12.8) -> tuple:
    """Calculate minimum unit cells for RASPA cutoff."""
    a = b = c = 25.0
    with open(cif_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("_cell_length_a"):
                a = float(line.split()[1].strip())
            elif line.startswith("_cell_length_b"):
                b = float(line.split()[1].strip())
            elif line.startswith("_cell_length_c"):
                c = float(line.split()[1].strip())
    na = max(1, math.ceil((2.0 * cutoff) / a))
    nb = max(1, math.ceil((2.0 * cutoff) / b))
    nc = max(1, math.ceil((2.0 * cutoff) / c))
    return na, nb, nc


def stage_zeo(cif_path: str, zeopp_bin: str, timeout: int = 120) -> tuple:
    """Zeo++ geometry calculation. Returns (success, geometry_dict, error_msg, seconds)."""
    t0 = time.time()
    cif_path = os.path.abspath(cif_path)
    cif_dir = os.path.dirname(cif_path)
    cif_stem = Path(cif_path).stem

    sa_out = os.path.join(cif_dir, f"{cif_stem}.sa")
    vol_out = os.path.join(cif_dir, f"{cif_stem}.vol")
    res_out = os.path.join(cif_dir, f"{cif_stem}.res")

    try:
        subprocess.run([zeopp_bin, "-sa", "1.2", "1.2", "5000", cif_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, check=False)
        subprocess.run([zeopp_bin, "-vol", "1.2", "1.2", "50000", cif_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, check=False)
        subprocess.run([zeopp_bin, "-res", cif_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, None, f"zeo++ timeout ({timeout}s)", time.time() - t0
    except Exception as e:
        return False, None, f"zeo++ error: {e}", time.time() - t0

    result: dict = {}

    if os.path.isfile(sa_out):
        with open(sa_out) as f:
            for line in f:
                if "ASA_m^2/cm^3:" in line:
                    try:
                        result["sa"] = float(line.split("ASA_m^2/cm^3:")[1].split()[0])
                        result["cv"] = float(line.split("Unitcell_volume:")[1].split()[0])
                        result["density"] = float(line.split("Density:")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
                    break

    if os.path.isfile(vol_out):
        with open(vol_out) as f:
            for line in f:
                if "AV_Volume_fraction:" in line:
                    try:
                        result["vf"] = float(line.split("AV_Volume_fraction:")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
                    break

    if os.path.isfile(res_out):
        with open(res_out) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        result["di"] = float(parts[1])
                        result["df"] = float(parts[2])
                        result["dif"] = float(parts[3])
                    except ValueError:
                        pass
                    break

    for tmp in [sa_out, vol_out, res_out]:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    if "di" not in result or "df" not in result:
        return False, None, "zeo++ missing pore size output", time.time() - t0

    return True, result, "", time.time() - t0


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


def _resolve_ads_cfg(adsorbate: str) -> dict:
    """Return adsorbate config dict from LIVE_SIM_ADSORBATE_CONFIGS, with H2 fallback."""
    if LIVE_SIM_ADSORBATE_CONFIGS:
        return LIVE_SIM_ADSORBATE_CONFIGS.get(adsorbate, LIVE_SIM_ADSORBATE_CONFIGS["h2"])
    # Hard fallback if config import failed
    return {
        "forcefield": "UFF_H2", "molecule": "hydrogen",
        "temperature": 77.0, "pressure": 10000000.0,
        "charge_method": "None", "mw_g_mol": 2.016, "xe_molfrac": None,
    }


def stage_raspa(cif_path: str, work_dir: str, cfg: dict, forcefield_dir: str,
                adsorbate: str = "h2", xe_molfrac: float = 0.20) -> tuple:
    """RASPA3 GCMC simulation. Returns (success, uptake_dict, error_msg, seconds)."""
    t0 = time.time()
    filename = Path(cif_path).stem
    timeout = cfg.get("raspa_timeout", 600)
    cutoff = 12.8

    ads_cfg = _resolve_ads_cfg(adsorbate)
    temperature = cfg.get("temperature") or ads_cfg["temperature"]
    pressure = cfg.get("pressure") or ads_cfg["pressure"]
    charge_method = ads_cfg["charge_method"]

    raspa_dir = os.path.join(work_dir, "raspa")
    os.makedirs(raspa_dir, exist_ok=True)
    shutil.copy(cif_path, raspa_dir)
    na, nb, nc = get_unit_cells(cif_path)

    system = {
        "Type": "Framework",
        "Name": filename,
        "NumberOfUnitCells": [na, nb, nc],
        "ExternalTemperature": temperature,
        "ExternalPressure": pressure,
        "ChargeMethod": charge_method,
    }
    if adsorbate == "co2":
        system["CutOffCoulomb"] = cutoff
        # NOTE: DDEC6 charges are pre-assigned by run_local_charge_stage() on local GPU
        # before R2 dispatch. _assign_framework_charges() is intentionally NOT called here.

    rotation_prob = 0.0 if adsorbate in ("xekr", "h2", "ch4") else 1.0
    swap_prob = 2.0 if pressure >= 3e6 else 1.0  # 30 bar = 3e6 Pa

    if adsorbate == "xekr":
        kr_molfrac = 1.0 - xe_molfrac
        components = []
        for name, mf in [("Xe", xe_molfrac), ("Kr", kr_molfrac)]:
            components.append({
                "Name": name,
                "MoleculeDefinition": os.path.join(forcefield_dir, f"{name}.json"),
                "MolFraction": mf,
                "FugacityCoefficient": 1.0,
                "IdealGasRosenbluthWeight": 1.0,
                "TranslationProbability": 1.0,
                "ReinsertionProbability": 1.0,
                "RotationProbability": rotation_prob,
                "SwapProbability": swap_prob,
                "WidomProbability": 1.0,
                "CreateNumberOfMolecules": 0,
            })
    else:
        molecule = ads_cfg["molecule"]
        components = [{
            "Name": molecule,
            "MoleculeDefinition": os.path.join(forcefield_dir, f"{molecule}.json"),
            "FugacityCoefficient": 1.0,
            "IdealGasRosenbluthWeight": 1.0,
            "TranslationProbability": 1.0,
            "ReinsertionProbability": 1.0,
            "RotationProbability": rotation_prob,
            "SwapProbability": swap_prob,
            "WidomProbability": 1.0,
            "CreateNumberOfMolecules": 0,
        }]

    sim_json = {
        "ForceField": forcefield_dir,
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": cfg.get("raspa_cycles", 1000),
        "NumberOfInitializationCycles": cfg.get("raspa_init_cycles", 500),
        "PrintEvery": 1000,
        "Systems": [system],
        "Components": components,
    }

    input_file = os.path.join(raspa_dir, "simulation.json")
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(sim_json, f, indent=2)

    raspa_bin = shutil.which("raspa3")
    if not raspa_bin:
        return False, None, "raspa3 not found", time.time() - t0

    env = os.environ.copy()
    env["RASPA_DIR"] = forcefield_dir
    log_file = os.path.join(raspa_dir, f"{filename}_raspa.log")

    try:
        with open(log_file, "w") as lf:
            result = subprocess.run(
                [raspa_bin, "simulation.json"],
                cwd=raspa_dir, env=env,
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        if result.returncode != 0:
            return False, None, f"raspa3 exit {result.returncode}", time.time() - t0
    except subprocess.TimeoutExpired:
        return False, None, f"raspa3 timeout ({timeout}s)", time.time() - t0

    # Parse output
    if adsorbate == "xekr":
        _parser = _parse_mixture if _parse_mixture is not None else _parse_mixture_inline
        uptake = _parser(raspa_dir, xe_molfrac=xe_molfrac, cif_path=cif_path)
        if not uptake or uptake.get("xe_loading_mol_kg") is None:
            return False, None, "No Xe/Kr loading in RASPA output", time.time() - t0
    else:
        uptake = parse_raspa_output(raspa_dir)
        if not uptake or not uptake.get("loading_mol_kg"):
            return False, None, "No loading data in RASPA output", time.time() - t0

    return True, uptake, "", time.time() - t0


def parse_raspa_output(raspa_dir: str) -> dict:
    """Parse RASPA3 output for H2 uptake.

    RASPA3 output format (near end of file):
      Abs. loading average   3.693e+01 +/-  7.323e-01 [mol/kg-framework]
      Excess loading average 2.416e+01 +/-  7.323e-01 [mol/kg-framework]
    Also handles inline format:
      3.505581e+01 mol/kg    (3.505581e+01 +/- 0.000000e+00)
    """
    output_dir = os.path.join(raspa_dir, "output")
    search_dir = output_dir if os.path.exists(output_dir) else raspa_dir

    output_file = None
    for f in os.listdir(search_dir):
        if f.startswith("output_") and f.endswith(".txt"):
            output_file = os.path.join(search_dir, f)
            break

    if not output_file:
        return {}

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    result = {}
    for line in lines:
        # RASPA3 summary format: "Abs. loading average   3.69e+01 +/-  7.32e-01 [mol/kg-framework]"
        if "Abs. loading average" in line and "mol/kg" in line:
            parts = line.split()
            try:
                idx = parts.index("average") + 1
                result["loading_mol_kg"] = float(parts[idx])
            except (ValueError, IndexError):
                pass

        elif "Excess loading average" in line and "mol/kg" in line:
            parts = line.split()
            try:
                idx = parts.index("average") + 1
                result["loading_excess_mol_kg"] = float(parts[idx])
            except (ValueError, IndexError):
                pass

        # RASPA3 Henry coefficient
        elif "Average Henry coefficient" in line and "mol/kg/Pa" in line:
            parts = line.split()
            try:
                idx = parts.index("coefficient:") + 1
                result["henry_mol_kg_pa"] = float(parts[idx])
            except (ValueError, IndexError):
                pass

        # Fallback: older RASPA2-style format
        elif "Average loading absolute [mol/kg]" in line:
            parts = line.split()
            for j, p in enumerate(parts):
                if p == "+/-":
                    try:
                        result["loading_mol_kg"] = float(parts[j - 1])
                    except (ValueError, IndexError):
                        pass
                    break

    return result


def main():
    args = parse_args()

    cfg, job = load_job(args.manifest, args.job_index)
    filename = job["filename"]
    pipeline = job.get("pipeline", "full")  # "full" | "stage1_only" | "stage2_only"
    adsorbate = args.adsorbate
    xe_molfrac = args.xe_molfrac
    ads_cfg = _resolve_ads_cfg(adsorbate)

    # CLI overrides for T/P (take precedence over manifest config and adsorbate defaults)
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.pressure is not None:
        cfg["pressure"] = args.pressure * 1e5  # bar → Pa

    print(f"[Job {args.job_index}] {filename} (beam={job['beam_id']}, pipeline={pipeline}, adsorbate={adsorbate})")

    work_dir = os.path.join(args.output_dir, filename)
    os.makedirs(work_dir, exist_ok=True)

    # Determine forcefield directory (auto-select from adsorbate config if not specified)
    ff_dir = args.forcefield_dir
    if not ff_dir:
        ff_name = ads_cfg["forcefield"]
        ff_dir = os.path.join(_proj_root, "core", "simulation", "gcmc", "forcefield", ff_name)
        if not os.path.exists(ff_dir):
            ff_dir = os.path.join(os.path.expanduser("~"), "llm4mof", "forcefields", ff_name)

    ff_dir = os.path.abspath(ff_dir)

    stages = {}
    result_data = {
        "filename": filename,
        "topology": job["topology"],
        "node": job["node"],
        "edge": job["edge"],
        "beam_id": job["beam_id"],
        "pipeline": pipeline,
        "predicted_geometry": job.get("predicted_geometry"),
        "match_score": job.get("match_score", 0.0),
        "status": "unknown",
        "cif_path": None,
        "real_uptake": None,
        "real_geometry": None,
        "error_msg": "",
        "wall_seconds": 0.0,
        "stages": {},
    }

    t_total = time.time()

    # -----------------------------------------------------------------------
    # Stage2-only: skip pormake/lammps/zeo, run RASPA with existing CIF
    # -----------------------------------------------------------------------
    if pipeline == "stage2_only":
        cif_path = job.get("cif_path")
        if not cif_path or not os.path.isfile(cif_path):
            result_data["status"] = "stage2_fail"
            result_data["error_msg"] = f"CIF not found for stage2: {cif_path}"
            result_data["wall_seconds"] = time.time() - t_total
            result_data["stages"] = stages
            _write_result(args.output_dir, filename, result_data)
            return

        # Carry forward real_geometry from stage1 (stored in job spec)
        real_geom = job.get("real_geometry") or None
        result_data["real_geometry"] = real_geom
        result_data["cif_path"] = cif_path

        ok, uptake, err, secs = stage_raspa(cif_path, work_dir, cfg, ff_dir, adsorbate, xe_molfrac)
        stages["raspa"] = {"status": "success" if ok else "fail", "seconds": secs, "error": err}

        if ok and uptake:
            result_data["status"] = "success"
            if adsorbate == "xekr":
                result_data["real_uptake"] = uptake
                xe = uptake.get("xe_loading_mol_kg", 0.0)
                kr = uptake.get("kr_loading_mol_kg", 0.0)
                sel = uptake.get("selectivity_xe_kr", float("nan"))
                print(f"   [OK] {filename}: Xe={xe:.3f} mol/kg, Kr={kr:.3f} mol/kg, "
                      f"S_Xe/Kr={sel:.2f} (stage2, zeo++ geometry)")
            else:
                mol_kg = float(uptake.get("loading_mol_kg", 0.0) or 0.0)
                density_for_gl = (
                    float(real_geom.get("density", 0.0) or 0.0) if real_geom
                    else float((result_data.get("predicted_geometry") or {}).get("density", 0.0) or 0.0)
                )
                mw = ads_cfg.get("mw_g_mol") or 2.016
                uptake["loading_g_L"] = round(mol_kg * density_for_gl * mw, 6)
                result_data["real_uptake"] = uptake
                gl = uptake["loading_g_L"]
                print(f"   [OK] {filename}: {adsorbate.upper()}={mol_kg} mol/kg, "
                      f"{gl:.2f} g/L (stage2, zeo++ geometry)")
        else:
            result_data["status"] = "raspa_fail"
            result_data["error_msg"] = err
            print(f"   [FAIL] {filename}: {err[:100]}")

        result_data["wall_seconds"] = time.time() - t_total
        result_data["stages"] = stages
        _write_result(args.output_dir, filename, result_data)
        return

    # -----------------------------------------------------------------------
    # Full or stage1_only: pormake → lammps → zeo [→ raspa]
    # -----------------------------------------------------------------------

    # Stage 1: PORMAKE
    ok, cif_path, err, secs = stage_pormake(job, work_dir)
    stages["pormake"] = {"status": "success" if ok else "fail", "seconds": secs, "error": err}
    if not ok:
        result_data["status"] = "build_fail"
        result_data["error_msg"] = err
        result_data["wall_seconds"] = time.time() - t_total
        result_data["stages"] = stages
        _write_result(args.output_dir, filename, result_data)
        return

    result_data["cif_path"] = cif_path

    # Stage 2: LAMMPS — returns optimized CIF path (or original on fallback)
    ok, active_cif_path, err, secs = stage_lammps(cif_path, work_dir, cfg)
    stages["lammps"] = {"status": "success" if ok else "fail", "seconds": secs, "error": err}
    if not ok:
        result_data["status"] = "lammps_fail"
        result_data["error_msg"] = err
        result_data["wall_seconds"] = time.time() - t_total
        result_data["stages"] = stages
        _write_result(args.output_dir, filename, result_data)
        return
    result_data["cif_path"] = active_cif_path

    # Stage 3: Zeo++ — always runs for stage1_only; optional for full (when --use-zeo)
    real_geom = None
    run_zeo = args.use_zeo or (pipeline == "stage1_only")
    if run_zeo:
        zeopp_bin = args.zeopp_bin
        if not zeopp_bin:
            zeopp_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network")
        if zeopp_bin and os.path.isfile(zeopp_bin):
            zok, real_geom, zerr, zsecs = stage_zeo(active_cif_path, zeopp_bin, timeout=300)
            stages["zeo"] = {"status": "success" if zok else "fail", "seconds": zsecs, "error": zerr}
            if not zok:
                result_data["status"] = "zeo_fail"
                result_data["error_msg"] = zerr
                result_data["wall_seconds"] = time.time() - t_total
                result_data["stages"] = stages
                _write_result(args.output_dir, filename, result_data)
                return
        else:
            stages["zeo"] = {"status": "fail", "seconds": 0.0, "error": f"zeopp binary not found: {zeopp_bin}"}
            result_data["status"] = "zeo_fail"
            result_data["error_msg"] = f"zeopp binary not found: {zeopp_bin}"
            result_data["wall_seconds"] = time.time() - t_total
            result_data["stages"] = stages
            _write_result(args.output_dir, filename, result_data)
            return

    result_data["real_geometry"] = real_geom

    # -----------------------------------------------------------------------
    # Stage1-only: stop here — no RASPA. Write result with cif_path + real_geometry.
    # -----------------------------------------------------------------------
    if pipeline == "stage1_only":
        result_data["status"] = "stage1_done"
        result_data["wall_seconds"] = time.time() - t_total
        result_data["stages"] = stages
        _write_result(args.output_dir, filename, result_data)
        df_val = real_geom.get("df") if real_geom else "N/A"
        print(f"   [STAGE1 DONE] {filename}: df={df_val}")
        return

    # -----------------------------------------------------------------------
    # Full pipeline: RASPA3
    # -----------------------------------------------------------------------
    if args.use_zeo and real_geom:
        geometry_filter = cfg.get("geometry_filter", {})
        if job["beam_id"] == "Z" and geometry_filter and not passes_geometry_filter(real_geom, geometry_filter):
            print(f"   [ZEO FILTER] {filename}: geometry not satisfied — proceeding to RASPA anyway")

    ok, uptake, err, secs = stage_raspa(active_cif_path, work_dir, cfg, ff_dir, adsorbate, xe_molfrac)
    stages["raspa"] = {"status": "success" if ok else "fail", "seconds": secs, "error": err}

    if ok and uptake:
        result_data["status"] = "success"
        if adsorbate == "xekr":
            result_data["real_uptake"] = uptake
            xe = uptake.get("xe_loading_mol_kg", 0.0)
            kr = uptake.get("kr_loading_mol_kg", 0.0)
            sel = uptake.get("selectivity_xe_kr", float("nan"))
            print(f"   [OK] {filename}: Xe={xe:.3f} mol/kg, Kr={kr:.3f} mol/kg, "
                  f"S_Xe/Kr={sel:.2f}" + (" (zeo++ geometry)" if real_geom else ""))
        else:
            mol_kg = float(uptake.get("loading_mol_kg", 0.0) or 0.0)
            # Compute loading_g_L: prefer zeo++ density, else predicted_geometry
            density_for_gl = (
                float(real_geom.get("density", 0.0) or 0.0) if real_geom
                else float((result_data.get("predicted_geometry") or {}).get("density", 0.0) or 0.0)
            )
            mw = ads_cfg.get("mw_g_mol") or 2.016
            uptake["loading_g_L"] = round(mol_kg * density_for_gl * mw, 6)
            result_data["real_uptake"] = uptake
            gl = uptake["loading_g_L"]
            print(f"   [OK] {filename}: {adsorbate.upper()}={mol_kg} mol/kg, {gl:.2f} g/L"
                  + (" (zeo++ geometry)" if real_geom else ""))
    else:
        result_data["status"] = "raspa_fail"
        result_data["error_msg"] = err
        print(f"   [FAIL] {filename}: {err[:100]}")

    result_data["wall_seconds"] = time.time() - t_total
    result_data["stages"] = stages
    _write_result(args.output_dir, filename, result_data)


def _write_result(output_dir: str, filename: str, data: dict) -> None:
    """Write per-MOF result JSON."""
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{filename}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Touch DONE sentinel
    done_path = os.path.join(output_dir, f"{filename}.DONE")
    with open(done_path, "w") as f:
        f.write("DONE\n")

    print(f"   [SAVED] {result_path}")


if __name__ == "__main__":
    main()
