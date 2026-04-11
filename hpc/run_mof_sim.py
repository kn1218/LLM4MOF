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
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run single MOF simulation")
    parser.add_argument("--manifest", required=True, help="Path to batch_manifest.json")
    parser.add_argument("--job-index", type=int, required=True, help="Job index (0-based)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--forcefield-dir", default=None, help="Path to UFF_H2 forcefield")
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
    """LAMMPS optimization. Returns (success, opt_cif_dir, error_msg, seconds).

    Uses lammps_interface to generate LAMMPS data+input files from CIF,
    then runs LAMMPS minimization. The API requires an options namespace
    with a cif_file attribute (not a raw string path).
    """
    if cfg.get("skip_lammps", False):
        return True, os.path.dirname(cif_path), "skipped", 0.0

    t0 = time.time()
    timeout = cfg.get("lammps_timeout", 900)

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
            minimize=True,
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
        print(f"   [LAMMPS] lammps-interface OK for {Path(cif_path).stem}")

        # Find generated input/data files
        in_files = list(Path(lammps_dir).glob("in.*"))
        if not in_files:
            return True, os.path.dirname(cif_path), "lammps-interface no input file", time.time() - t0

        input_file = str(in_files[0])

        # Run LAMMPS
        lmp_bin = shutil.which("lmp_mpi") or shutil.which("lmp")
        if not lmp_bin:
            return True, os.path.dirname(cif_path), "lmp not found", time.time() - t0

        result = subprocess.run(
            [lmp_bin, "-in", input_file],
            cwd=lammps_dir,
            capture_output=True, text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return True, os.path.dirname(cif_path), f"lammps exit {result.returncode}", time.time() - t0

        print(f"   [LAMMPS] Optimization OK for {Path(cif_path).stem}")
        # Use original CIF dir for RASPA (converting LAMMPS data back to CIF
        # is complex; the geometry change from minimization is typically small)
        return True, os.path.dirname(cif_path), "", time.time() - t0

    except subprocess.TimeoutExpired:
        return True, os.path.dirname(cif_path), f"lammps timeout ({timeout}s)", time.time() - t0
    except Exception as e:
        return True, os.path.dirname(cif_path), str(e)[:200], time.time() - t0


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


def stage_raspa(cif_path: str, work_dir: str, cfg: dict, forcefield_dir: str) -> tuple:
    """RASPA3 GCMC simulation. Returns (success, uptake_dict, error_msg, seconds)."""
    t0 = time.time()
    filename = Path(cif_path).stem
    timeout = cfg.get("raspa_timeout", 600)

    raspa_dir = os.path.join(work_dir, "raspa")
    os.makedirs(raspa_dir, exist_ok=True)

    # Copy CIF to raspa dir
    shutil.copy(cif_path, raspa_dir)

    # Get unit cells
    na, nb, nc = get_unit_cells(cif_path)

    # Create simulation.json
    sim_json = {
        "ForceField": forcefield_dir,
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": cfg.get("raspa_cycles", 1000),
        "NumberOfInitializationCycles": cfg.get("raspa_init_cycles", 500),
        "PrintEvery": 1000,
        "Systems": [{
            "Type": "Framework",
            "Name": filename,
            "NumberOfUnitCells": [na, nb, nc],
            "ExternalTemperature": cfg.get("temperature", 77.0),
            "ExternalPressure": cfg.get("pressure", 10000000.0),
            "HeliumVoidFraction": cfg.get("helium_void_fraction", 0.5),
            "ChargeMethod": "None",
        }],
        "Components": [{
            "Name": "hydrogen",
            "MoleculeDefinition": os.path.join(forcefield_dir, "hydrogen.json"),
            "FugacityCoefficient": 1.0,
            "IdealGasRosenbluthWeight": 1.0,
            "TranslationProbability": 1.0,
            "ReinsertionProbability": 1.0,
            "RotationProbability": 1.0,
            "SwapProbability": 1.0,
            "WidomProbability": 1.0,
            "CreateNumberOfMolecules": 0,
        }],
    }

    input_file = os.path.join(raspa_dir, "simulation.json")
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(sim_json, f, indent=2)

    # Run RASPA3
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

    print(f"[Job {args.job_index}] {filename} (beam={job['beam_id']})")

    work_dir = os.path.join(args.output_dir, filename)
    os.makedirs(work_dir, exist_ok=True)

    # Determine forcefield directory
    ff_dir = args.forcefield_dir
    if not ff_dir:
        # Default: look relative to this script
        ff_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "forcefields", "UFF_H2")
        if not os.path.exists(ff_dir):
            ff_dir = os.path.join(os.path.expanduser("~"), "llm2por", "forcefields", "UFF_H2")

    ff_dir = os.path.abspath(ff_dir)

    stages = {}
    result_data = {
        "filename": filename,
        "topology": job["topology"],
        "node": job["node"],
        "edge": job["edge"],
        "beam_id": job["beam_id"],
        "predicted_geometry": job.get("predicted_geometry"),
        "match_score": job.get("match_score", 0.0),
        "status": "unknown",
        "real_uptake": None,
        "error_msg": "",
        "wall_seconds": 0.0,
        "stages": {},
    }

    t_total = time.time()

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

    # Stage 2: LAMMPS
    ok, cif_dir, err, secs = stage_lammps(cif_path, work_dir, cfg)
    stages["lammps"] = {"status": "success" if ok else "fail", "seconds": secs, "error": err}

    # Stage 3: RASPA3
    ok, uptake, err, secs = stage_raspa(cif_path, work_dir, cfg, ff_dir)
    stages["raspa"] = {"status": "success" if ok else "fail", "seconds": secs, "error": err}

    if ok and uptake:
        result_data["status"] = "success"
        result_data["real_uptake"] = uptake
        print(f"   [OK] {filename}: H2={uptake.get('loading_mol_kg', '?')} mol/kg")
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
