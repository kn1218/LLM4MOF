#!/home/users/seunghh/anaconda3/envs/llm2por/bin/python
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from core import __root_dir__
from core.simulation.gcmc.raspa_utils import get_density_from_cif, parse_output

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "raspa_results",
)


def analyze_directory(output_dir):
    print(f"Analyzing RASPA results in: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"[ERROR] Directory not found: {output_dir}")
        return

    results = []

    for mof_dir in sorted(os.listdir(output_dir)):
        work_dir = os.path.join(output_dir, mof_dir)
        if not os.path.isdir(work_dir):
            continue

        # Skip if no output directory
        if not os.path.exists(os.path.join(work_dir, "output")):
            continue

        # Try to parse output (pass CIF path for density calculation)
        cif_file = os.path.join(work_dir, f"{mof_dir}.cif")
        raspa_result = parse_output(
            work_dir, cif_path=cif_file if os.path.exists(cif_file) else None
        )
        if raspa_result is None:
            print(f"  [WARNING] No valid output found for {mof_dir}")
            continue

        results.append({"mof": mof_dir, "raspa_result": raspa_result})

    print(f"\n[Results summary]")
    print("-" * 85)
    print(
        f"{'MOF':<25} {'Loading (mol/kg)':<18} {'Loading (mg/g)':<15} {'Loading (g/L)':<15} {'Density':<10}"
    )
    print("-" * 85)

    for r in results:
        mof = r["mof"]
        raspa = r["raspa_result"]
        loading_mol = raspa.get("loading_mol_kg", 0)
        loading_mg = raspa.get("loading_mg_g", 0)
        loading_gl = raspa.get("loading_g_L", 0)
        density = raspa.get("framework_density_g_cm3", 0)

        # Format safely
        density_str = f"{density:.2f}" if density else "N/A"
        loading_gl_str = f"{loading_gl:.2f}" if density else "N/A"

        print(
            f"{mof:<25} {loading_mol:<18.2f} {loading_mg:<15.2f} {loading_gl_str:<15} {density_str:<10}"
        )

    results_file = os.path.join(output_dir, "simulation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RASPA3 GCMC results")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory to analyze"
    )
    args = parser.parse_args()
    analyze_directory(args.output_dir)


if __name__ == "__main__":
    main()
