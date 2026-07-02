"""
Run RASPA3 GCMC simulations for top 20 MOFs from test_result_agent3.json
Uses RASPA3 JSON format

Usage:
    python run_raspa.py [--mof-dir DIR] [--output-dir DIR]

Requirements:
    - CIF files for each MOF in mof-dir (naming: topology+node+edge.cif)
    - RASPA3 installed (conda install raspa3 -c conda-forge)
    - UFF_H2 forcefield in RASPA3 JSON format
"""

import os
import sys
import math
import json
import argparse
import subprocess
import shutil
import importlib.util
from pathlib import Path
from typing import Optional


def find_raspa3():
    """Find raspa3 binary automatically."""
    # Try to find from conda/mamba environment
    raspa_spec = importlib.util.find_spec("raspa")
    if raspa_spec:
        # Try to get the executable path from the raspa module
        raspa_dir = os.path.dirname(raspa_spec.submodule_search_locations[0])
        raspa_bin = os.path.join(raspa_dir, "bin", "raspa3")
        if os.path.exists(raspa_bin):
            return raspa_bin

    # Fallback: try which
    raspa_which = shutil.which("raspa3")
    if raspa_which:
        return raspa_which

    # Fallback: try conda env directories (Scripts/, bin/, Library/bin/)
    env_root = os.path.dirname(sys.executable)
    for subdir in ["", "bin", "Scripts", "Library/bin"]:
        for ext in ["", ".exe"]:
            candidate = os.path.join(env_root, subdir, f"raspa3{ext}")
            if os.path.exists(candidate):
                return candidate

    # Also try one level up (sys.executable may be in Scripts/ or bin/)
    env_root_parent = os.path.dirname(env_root)
    for subdir in ["bin", "Scripts", "Library/bin"]:
        for ext in ["", ".exe"]:
            candidate = os.path.join(env_root_parent, subdir, f"raspa3{ext}")
            if os.path.exists(candidate):
                return candidate

    return None


# Resolve the project root robustly. __file__ is .../core/simulation/gcmc/run_raspa.py,
# so 4x dirname reaches the repo root. Using fewer levels lands on .../core/ and can make
# `from core import __root_dir__` fail or pick up a stale `core` package from another
# project on sys.path:
#   __file__              = .../core/simulation/gcmc/run_raspa.py
#   dirname x1            = .../core/simulation/gcmc/
#   dirname x2            = .../core/simulation/
#   dirname x3            = .../core/             (too shallow -- wrong)
#   dirname x4            = .../                  (project root -- correct)
_project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
sys.path.insert(0, _project_root)
from core import __root_dir__
from core.simulation.gcmc.raspa_utils import get_density_from_cif

DEFAULT_RASPA3 = find_raspa3() or shutil.which("raspa3")

RESULT_FILE = os.path.join(__root_dir__, "..", "test_result_agent3.json")
DEFAULT_MOF_DIR = os.path.join(__root_dir__, "..", "..", "cif")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "raspa_results",
)
DEFAULT_FORCEFIELD_DIR = os.path.join(
    __root_dir__, "simulation", "gcmc", "forcefield", "UFF_H2"
)
FORCEFIELD_BASE_DIR = os.path.join(__root_dir__, "simulation", "gcmc", "forcefield")

# Per-adsorbate simulation defaults
ADSORBATE_CONFIGS = {
    "h2": {
        "forcefield": "UFF_H2",
        "molecule": "hydrogen",
        "temperature": 77.0,
        "pressure": 1e7,
        "charge_method": "None",
    },
    "ch4": {
        "forcefield": "UFF",
        "molecule": "CH4",
        "temperature": 298.0,
        "pressure": 2.5e5,    # 2.5 bar
        "charge_method": "None",
    },
    "co2": {
        "forcefield": "UFF",
        "molecule": "CO2",
        "temperature": 298.0,
        "pressure": 2.5e5,    # 2.5 bar
        "charge_method": "Ewald",
    },
    "xekr": {
        "forcefield": "UFF_XeKr",
        "molecule": None,     # 2-component mixture
        "temperature": 273.0,
        "pressure": 1e5,      # 1 bar
        "charge_method": "None",
    },
}


def load_mof_data_from_json():
    """Load top 20 MOFs from test_result_agent3.json"""
    with open(RESULT_FILE, "r") as f:
        data = json.load(f)
    return data["proposals"]["ranked_mofs"][:20]


def load_mof_data_from_cif_dir(cif_dir):
    mofs = []
    for f in os.listdir(cif_dir):
        if f.endswith(".cif"):
            filename = f[:-4]
            mofs.append(
                {"filename": filename, "rank": len(mofs) + 1, "predicted_geometry": {}}
            )
    return sorted(mofs, key=lambda x: x["filename"])


def get_unit_cells(cif_path, cutoff=12.8):
    """
    Calculate the minimum number of unit cells needed to satisfy cutoff radius.

    Args:
        cif_path: Path to CIF file
        cutoff: Cutoff radius for LJ interactions (default 12.8 Angstrom)

    Returns:
        Tuple of (na, nb, nc) - number of unit cells in each direction
    """
    a, b, c = 10.0, 10.0, 10.0  # defaults if not found

    with open(cif_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("_cell_length_a"):
                a = float(line.split()[1].strip())
            elif line.startswith("_cell_length_b"):
                b = float(line.split()[1].strip())
            elif line.startswith("_cell_length_c"):
                c = float(line.split()[1].strip())

    na = math.ceil((2.0 * cutoff) / a)
    nb = math.ceil((2.0 * cutoff) / b)
    nc = math.ceil((2.0 * cutoff) / c)

    return max(1, na), max(1, nb), max(1, nc)


def assign_framework_charges(cif_path: str) -> None:
    """Assign DDEC6 partial atomic charges to framework CIF using PACMAN-charge (in-place).

    Runs PACMAN-charge GNN to predict DDEC6 charges, produces {stem}_pacman.cif,
    then renames it back to the original cif_path so downstream code is unaffected.
    Silently skips if PACMAN-charge is not installed or prediction fails.
    """
    try:
        from PACMANCharge import pmcharge
    except ImportError:
        print(f"   [CHARGE] PACMAN-charge not installed; skipping charge assignment")
        print(f"   [CHARGE] To install: pip install PACMAN-charge")
        return

    pacman_cif = cif_path.replace(".cif", "_pacman.cif")
    try:
        pmcharge.predict(
            cif_file=cif_path,
            charge_type="DDEC6",
            digits=6,
            atom_type=True,
            neutral=True,
            keep_connect=True,
        )
        if os.path.isfile(pacman_cif):
            os.replace(pacman_cif, cif_path)
            print(f"   [CHARGE] DDEC6 charges assigned: {os.path.basename(cif_path)}")
        else:
            print(f"   [CHARGE] PACMAN output not found; using original CIF")
    except Exception as e:
        print(f"   [CHARGE] PACMAN-charge failed ({e}); using original CIF")
        if os.path.isfile(pacman_cif):
            os.remove(pacman_cif)


def _build_single_component(forcefield_path: str, molecule: str) -> dict:
    """Build a single-component GCMC component dict for RASPA3 JSON."""
    return {
        "Name": molecule,
        "MoleculeDefinition": os.path.join(forcefield_path, f"{molecule}.json"),
        "FugacityCoefficient": 1.0,
        "IdealGasRosenbluthWeight": 1.0,
        "TranslationProbability": 1.0,
        "ReinsertionProbability": 1.0,
        "RotationProbability": 1.0,
        "SwapProbability": 1.0,
        "WidomProbability": 1.0,
        "CreateNumberOfMolecules": 0,
    }


def _build_xekr_components(forcefield_path: str, xe_molfrac: float = 0.20) -> list:
    """Build 2-component Xe/Kr mixture component list for RASPA3 JSON (CBMC identity swap)."""
    kr_molfrac = 1.0 - xe_molfrac
    components = []
    for name, molfrac in [("Xe", xe_molfrac), ("Kr", kr_molfrac)]:
        components.append({
            "Name": name,
            "MoleculeDefinition": os.path.join(forcefield_path, f"{name}.json"),
            "MolFraction": molfrac,
            "FugacityCoefficient": 1.0,
            "IdealGasRosenbluthWeight": 1.0,
            "TranslationProbability": 1.0,
            "ReinsertionProbability": 1.0,
            "RotationProbability": 1.0,
            "SwapProbability": 1.0,
            "WidomProbability": 1.0,
            "CreateNumberOfMolecules": 0,
        })
    return components


def create_raspa_input(mof, mof_dir, output_dir, params):
    """Create RASPA3 JSON input file for a single MOF."""
    filename = mof["filename"]
    adsorbate = params.get("adsorbate", "h2")
    cfg = ADSORBATE_CONFIGS[adsorbate]

    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(output_dir, filename, "simulation.json")
    os.makedirs(os.path.dirname(input_file), exist_ok=True)

    forcefield_path = os.path.abspath(
        params.get("forcefield_dir") or os.path.join(FORCEFIELD_BASE_DIR, cfg["forcefield"])
    )

    cutoff = params.get("cutoff", 12.8)
    cif_file = os.path.join(mof_dir, f"{filename}.cif")
    na, nb, nc = get_unit_cells(cif_file, cutoff)
    density = get_density_from_cif(cif_file)

    temperature = params.get("temperature") or cfg["temperature"]
    pressure = params.get("pressure") or cfg["pressure"]

    # CO2: assign framework charges (DDEC stub — not yet implemented)
    if adsorbate == "co2":
        assign_framework_charges(cif_file)

    system = {
        "Type": "Framework",
        "Name": filename,
        "NumberOfUnitCells": [na, nb, nc],
        "ExternalTemperature": temperature,
        "ExternalPressure": pressure,
        "ChargeMethod": cfg["charge_method"],
    }
    if adsorbate == "co2":
        system["CutOffCoulomb"] = cutoff

    if adsorbate == "xekr":
        components = _build_xekr_components(
            forcefield_path, xe_molfrac=params.get("xe_molfrac", 0.20)
        )
    else:
        components = [_build_single_component(forcefield_path, cfg["molecule"])]

    rotation_prob = 0.0 if adsorbate in ("xekr", "h2", "ch4") else 1.0
    swap_prob = 2.0 if pressure >= 3e6 else 1.0  # 30 bar = 3e6 Pa
    for comp in components:
        comp["RotationProbability"] = rotation_prob
        comp["SwapProbability"] = swap_prob

    simulation_json = {
        "ForceField": forcefield_path,
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": params.get("cycles", 10000),
        "NumberOfInitializationCycles": params.get("init_cycles", 5000),
        "PrintEvery": 1000,
        "Systems": [system],
        "Components": components,
    }

    with open(input_file, "w") as f:
        json.dump(simulation_json, f, indent=2)

    return input_file


def run_simulation_background(
    input_file, mof_dir, output_dir, forcefield_dir, filename, raspa3_bin,
    timeout=600,
):
    """
    Run RASPA3 simulation in background using nohup.
    Returns immediately after launching.
    """
    cif_file = os.path.join(mof_dir, f"{filename}.cif")

    if not os.path.exists(cif_file):
        print(f"   [WARNING] CIF file not found: {cif_file}")
        return False

    work_dir = os.path.dirname(input_file)

    shutil.copy(cif_file, work_dir)

    env = os.environ.copy()
    env["RASPA_DIR"] = forcefield_dir

    # Create log file path
    log_file = os.path.join(work_dir, f"{filename}_raspa.log")
    done_file = os.path.join(work_dir, f"{filename}.DONE.txt")

    # Check if already done
    if os.path.exists(done_file):
        print(f"   [SKIP] {filename} - already completed")
        return True

    # FIX 2026-04-09: the original used a bash-style "cd && raspa3 ..." shell
    # string via subprocess.Popen(shell=True). On Windows that becomes
    # `cmd.exe /c "cd <path> && raspa3 simulation.json > log 2>&1 && ..."`
    # which fails with "The system cannot find the path specified" because of
    # cmd.exe path-parsing quirks (especially with `+` chars in work_dir names).
    # Replace with the portable subprocess.run + cwd= pattern. This is also
    # SYNCHRONOUS, which is fine because the surrounding launcher already has
    # a "wait for DONE files" loop after the launch phase.
    print(f"   [LAUNCH] {filename} via subprocess.run cwd={work_dir}")
    try:
        with open(log_file, "w") as lf:
            result = subprocess.run(
                [raspa3_bin, "simulation.json"],
                cwd=work_dir,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                shell=False,
                timeout=timeout,
            )
        if result.returncode == 0:
            with open(done_file, "w") as df:
                df.write("DONE\n")
            print(f"   [DONE] {filename}")
            return True
        else:
            print(f"   [ERROR] {filename} raspa3 exit {result.returncode} (see {log_file})")
            return False
    except subprocess.TimeoutExpired:
        print(f"   [TIMEOUT] {filename} exceeded {timeout}s cap")
        return False
    except Exception as e:
        print(f"   [ERROR] Failed to run {filename}: {e}")
        return False


def check_simulation_complete(input_file, filename):
    """Check if simulation is complete for a single MOF."""
    work_dir = os.path.dirname(input_file)
    done_file = os.path.join(work_dir, f"{filename}.DONE.txt")
    return os.path.exists(done_file)


def check_all_complete(mofs, output_dir):
    """Check if all simulations are complete."""
    for mof in mofs:
        filename = mof["filename"]
        mof_output_dir = os.path.join(output_dir, filename)
        done_file = os.path.join(mof_output_dir, f"{filename}.DONE.txt")
        if not os.path.exists(done_file):
            return False
    return True


def wait_for_simulations(mofs, output_dir, check_interval=60):
    """Wait for all RASPA3 simulations to complete."""
    import time

    print(f"\n[Wait] Monitoring simulations (check every {check_interval}s)...")

    while True:
        all_complete = True
        running = 0

        for mof in mofs:
            filename = mof["filename"]
            mof_output_dir = os.path.join(output_dir, filename)
            done_file = os.path.join(mof_output_dir, f"{filename}.DONE.txt")

            if not os.path.exists(done_file):
                all_complete = False
                running += 1

        if all_complete:
            print(f"[Done] All {len(mofs)} simulations completed!")
            return True

        print(f"[Wait] {running}/{len(mofs)} still running...")
        time.sleep(check_interval)


def run_simulation(input_file, mof_dir, output_dir, forcefield_dir, raspa3_bin):
    """Run RASPA3 simulation (CPU only)"""
    filename = os.path.basename(os.path.dirname(input_file))
    cif_file = os.path.join(mof_dir, f"{filename}.cif")

    if not os.path.exists(cif_file):
        print(f"   [WARNING] CIF file not found: {cif_file}")
        return None

    work_dir = os.path.dirname(input_file)

    shutil.copy(cif_file, work_dir)

    env = os.environ.copy()
    env["RASPA_DIR"] = forcefield_dir

    env["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)
    print(
        f"   Running RASPA3 for {filename} [CPU: {env['OMP_NUM_THREADS']} threads]..."
    )

    cmd = [raspa3_bin, "simulation.json"]
    subprocess.run(
        cmd,
        shell=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=work_dir,
    )

    return True


def parse_output(output_dir):
    output_subdir = os.path.join(output_dir, "output")
    output_file = None

    search_dir = output_subdir if os.path.exists(output_subdir) else output_dir

    for f in os.listdir(search_dir):
        if f.startswith("output_") and f.endswith(".txt"):
            output_file = os.path.join(search_dir, f)
            break

    if not output_file or not os.path.exists(output_file):
        return None

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    result = {}
    final_section = False
    final_results = {}
    initial_results = {}
    last_abs_molecules = None
    mg_g_final = None

    unit_cell_a = unit_cell_b = unit_cell_c = None
    framework_mass_amu = None
    num_unit_cells = 2

    for i, line in enumerate(lines):
        if "Framework" in line and unit_cell_a is None:
            for j in range(i, min(i + 15, len(lines))):
                if "Lengths:" in lines[j]:
                    parts = lines[j].split()
                    try:
                        unit_cell_a = float(parts[1])
                        unit_cell_b = float(parts[2])
                        unit_cell_c = float(parts[3])
                    except:
                        pass
                if "mass:" in lines[j] and "amu" in lines[j]:
                    parts = lines[j].split()
                    try:
                        framework_mass_amu = float(parts[1])
                    except:
                        pass

        if "NumberOfUnitCells" in line and num_unit_cells == 2:
            parts = line.split()
            try:
                num_unit_cells = int(parts[1])
            except:
                pass

        if (
            "mass:" in line
            and "amu" in line
            and "[-]" not in line
            and framework_mass_amu is None
        ):
            parts = line.split()
            try:
                framework_mass_amu = float(parts[1])
            except:
                pass

        if (
            "mass:" in line
            and "amu" in line
            and "[-]" not in line
            and framework_mass_amu is None
        ):
            parts = line.split()
            try:
                framework_mass_amu = float(parts[1])
            except:
                pass

        if "Final state after" in line or "Simulation finished" in line:
            final_section = True

        if "absolute adsorption:" in line and "molecules" in line:
            parts = line.split()
            try:
                last_abs_molecules = float(parts[2])
            except:
                pass

        if "mol/kg" in line and not line.strip().startswith("Block"):
            try:
                mol_kg = float(line.split()[0])
                if final_section:
                    final_results["loading_mol_kg"] = mol_kg
                    if last_abs_molecules:
                        final_results["loading_molecules"] = last_abs_molecules
                else:
                    initial_results["loading_mol_kg"] = mol_kg
                    if last_abs_molecules:
                        initial_results["loading_molecules"] = last_abs_molecules
            except:
                pass

        if "mg/g" in line and "[-]" not in line:
            try:
                mg_g = float(line.split()[0])
                mg_g_final = mg_g
            except:
                pass

    if final_results:
        result.update(final_results)
    elif initial_results:
        result.update(initial_results)

    if result.get("loading_mol_kg"):
        result["loading_g_L"] = result["loading_mol_kg"] * 2.016

    if mg_g_final:
        result["loading_mg_g"] = mg_g_final

    return result if result else None


def main():
    parser = argparse.ArgumentParser(description="Run RASPA3 GCMC for top 20 MOFs")
    parser.add_argument(
        "--mof-dir", default=DEFAULT_MOF_DIR, help="Directory with CIF files"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--forcefield-dir",
        default=None,
        help="Force field directory (JSON format); auto-selected from --adsorbate if not set",
    )
    parser.add_argument(
        "--adsorbate",
        default="h2",
        choices=list(ADSORBATE_CONFIGS.keys()),
        help="Adsorbate type: h2 | ch4 | co2 | xekr (default: h2)",
    )
    parser.add_argument(
        "--xe-molfrac",
        type=float,
        default=0.20,
        help="Xe mole fraction for xekr mixture (default: 0.20)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Temperature (K); defaults to adsorbate config value",
    )
    parser.add_argument(
        "--pressure", type=float, default=None,
        help="Pressure (Pa); defaults to adsorbate config value",
    )
    parser.add_argument(
        "--cutoff", type=float, default=12.8, help="Cutoff radius (Angstrom)"
    )
    parser.add_argument("--cycles", type=int, default=10000, help="Number of MC cycles")
    parser.add_argument(
        "--init-cycles", type=int, default=5000, help="Number of initialization cycles"
    )
    parser.add_argument(
        "--raspa3", default=DEFAULT_RASPA3, help="Path to raspa3 binary"
    )

    args = parser.parse_args()

    cfg = ADSORBATE_CONFIGS[args.adsorbate]
    temperature = args.temperature or cfg["temperature"]
    pressure = args.pressure or cfg["pressure"]

    print("=" * 70)
    print("RASPA3 GCMC Simulation Runner (JSON Format)")
    print("=" * 70)

    print(f"\n[1] Loading MOFs from CIF directory: {args.mof_dir}")
    mofs = load_mof_data_from_cif_dir(args.mof_dir)
    print(f"    Found {len(mofs)} CIF files")

    print(f"\n[2] Simulation parameters:")
    print(f"    Adsorbate: {args.adsorbate}")
    print(f"    Temperature: {temperature} K")
    print(f"    Pressure: {pressure} Pa ({pressure / 1e5:.1f} bar)")
    print(f"    ChargeMethod: {cfg['charge_method']}")
    print(f"    Cutoff: {args.cutoff} Angstrom")
    print(f"    Forcefield: {args.forcefield_dir or cfg['forcefield']} (auto)")
    print(f"    Cycles: {args.cycles}")
    if args.adsorbate == "xekr":
        print(f"    Xe mole fraction: {args.xe_molfrac:.2f}  Kr: {1 - args.xe_molfrac:.2f}")

    params = {
        "adsorbate": args.adsorbate,
        "temperature": args.temperature,  # None → use cfg default inside create_raspa_input
        "pressure": args.pressure,
        "cycles": args.cycles,
        "init_cycles": args.init_cycles,
        "forcefield_dir": args.forcefield_dir,
        "cutoff": args.cutoff,
        "xe_molfrac": args.xe_molfrac,
    }

    print(f"\n[3] Checking CIF files in: {args.mof_dir}")
    cif_count = 0
    for mof in mofs:
        cif_file = os.path.join(args.mof_dir, f"{mof['filename']}.cif")
        if os.path.exists(cif_file):
            cif_count += 1

    print(f"    Found {cif_count}/{len(mofs)} CIF files")

    if cif_count == 0:
        print("\n[WARNING] No CIF files found!")
        print(f"    Please place CIF files in: {args.mof_dir}")
        print("    Naming convention: topology+node+edge.cif")
        print("    Example: qzd+N32+E17.cif")
        return

    print(f"\n[4] Starting RASPA3 simulations (background)...")

    for i, mof in enumerate(mofs):
        filename = mof["filename"]
        cif_file = os.path.join(args.mof_dir, f"{filename}.cif")
        na, nb, nc = get_unit_cells(cif_file, args.cutoff)
        density = get_density_from_cif(cif_file)

        print(f"\n    [{i + 1}/{len(mofs)}] {filename} (UnitCells: {na}x{nb}x{nc})")

        input_file = create_raspa_input(mof, args.mof_dir, args.output_dir, params)

        forcefield_dir_resolved = args.forcefield_dir or os.path.join(
            FORCEFIELD_BASE_DIR, cfg["forcefield"]
        )
        success = run_simulation_background(
            input_file,
            args.mof_dir,
            args.output_dir,
            forcefield_dir_resolved,
            filename,
            args.raspa3,
        )

        if success:
            print(f"       [LAUNCHED] {filename} started in background")
        else:
            print(f"       [FAILED] Could not launch {filename}")

    print("\n" + "=" * 70)
    print("All simulations launched. Waiting for completion...")

    wait_for_simulations(mofs, args.output_dir, check_interval=60)

    print("\n[5] Analyzing results...")
    from core.simulation.gcmc.analyze import analyze_directory

    analyze_directory(args.output_dir)

    print("\n" + "=" * 70)
    print("ALL SIMULATIONS COMPLETED AND ANALYZED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
