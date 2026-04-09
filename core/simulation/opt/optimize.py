"""
Molecular Simulation Laboratory
LAMMPS Optimization Pipeline for MOF CIF files.

Usage:
    python core/opt/optimize.py --cif-dir /path/to/cifs
    python core/opt/optimize.py --cif-dir /path/to/cifs --run-lammps
    python core/opt/optimize.py --convert-only --cif-dir /path/to/cifs
"""

# =============================================================================
# Module Import
# =============================================================================
import os
import sys
import re
import argparse
from pathlib import Path
from itertools import tee
from typing import Optional, List, Tuple

# Add parent directory to path for imports
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root_dir)
__root_dir__ = _root_dir

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_CIF_DIR = os.path.join(__root_dir__, "..", "cif")
DEFAULT_OUTPUT_DIR = os.path.join(__root_dir__, "..", "experiments", "opt_output")

# LAMMPS settings (adjust as needed)
LAMMPS_EXECUTABLE = os.getenv("LAMMPS_EXECUTABLE", "lmp_mpi")
LAMMPS_THREADS = os.getenv("OMP_NUM_THREADS", "1")

# Converter settings
CONVERTER_SCRIPT = os.path.join(__root_dir__, "opt", "converter.py")


# =============================================================================
# Console Output
# =============================================================================
def log_info(msg: str) -> None:
    print(f"[Optimize] {msg}")


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")


def log_step(step: str, msg: str) -> None:
    print(f"[{step}] {msg}")


# =============================================================================
# LAMMPS Interface Step
# =============================================================================
def run_lammps_interface(
    cif_file: Path, output_dir: Path, lammps_interface_cmd: str = "lammps-interface"
) -> bool:
    """
    Run lammps-interface on a CIF file to generate LAMMPS data file.

    Args:
        cif_file: Path to input CIF file
        output_dir: Directory to save output
        lammps_interface_cmd: Command for lammps-interface

    Returns:
        True if successful, False otherwise
    """
    output_file = output_dir / f"data.{cif_file.stem}"

    # Check for original and optimized files
    opt_output_file = output_dir / f"data.{cif_file.stem}_opt"

    if output_file.exists():
        log_info(f"Skipping {cif_file.name} - output already exists")
        return True

    try:
        import subprocess

        cmd = [lammps_interface_cmd, str(cif_file.absolute())]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=str(output_dir)
        )

        if result.returncode != 0:
            log_error(f"lammps-interface failed for {cif_file.name}: {result.stderr}")
            return False

        if not output_file.exists():
            log_error(f"lammps-interface did not generate data.{cif_file.stem}")
            return False

        log_info(f"Generated {output_file.name}")
        return True

    except subprocess.TimeoutExpired:
        log_error(f"lammps-interface timed out for {cif_file.name}")
        return False
    except Exception as e:
        log_error(f"Error running lammps-interface for {cif_file.name}: {e}")
        return False


def make_lammps_interface_jobs(
    cif_dir: Path, output_dir: Path, lammps_interface_cmd: str = "lammps-interface"
) -> None:
    """
    Process all CIF files in directory with lammps-interface.

    Args:
        cif_dir: Directory containing CIF files
        output_dir: Directory to save LAMMPS data files
        lammps_interface_cmd: Command for lammps-interface
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    cif_files = list(cif_dir.glob("*.cif"))
    if not cif_files:
        log_error(f"No CIF files found in {cif_dir}")
        return

    log_step("LAMMPS-Interface", f"Processing {len(cif_files)} CIF files...")

    success_count = 0
    for cif_file in cif_files:
        if run_lammps_interface(cif_file, output_dir, lammps_interface_cmd):
            success_count += 1

    log_step("LAMMPS-Interface", f"Completed {success_count}/{len(cif_files)} files")


# =============================================================================
# LAMMPS Optimization Step
# =============================================================================
LAMMPS_MINIMIZE_EXE = """
thermo 1000

fix llm2por_relax all box/relax iso 0.0 vmax 0.001
minimize 1.0e-4 1.0e-6 5000 50000
unfix llm2por_relax

run 0

compute llm2por_pe all pe
variable llm2por_A equal c_llm2por_pe

thermo_style custom temp c_llm2por_pe
thermo 1
run 0

print "{{{{}}}}:${{llm2por_A}}" append {{0}}_energy.txt

write_data           {1}

uncompute llm2por_pe
variable llm2por_A delete
"""


def make_optimization_input(data_dir: Path, output_dir: Path) -> List[Path]:
    """
    Create LAMMPS input files for optimization by appending to lammps-interface output.

    Args:
        data_dir: Directory containing LAMMPS data files
        output_dir: Directory to save input files

    Returns:
        List of created input file paths
    """
    input_dir = output_dir / "in_files"
    log_dir = output_dir / "log"
    out_dir = output_dir / "opt_lammps_data"

    input_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Use existing in.* files from lammps-interface
    in_files = list(data_dir.glob("in.*"))
    if not in_files:
        log_error(f"No in files found in {data_dir}")
        return []

    created_files = []
    for in_file in in_files:
        cif_id = in_file.name.replace("in.", "")

        out_file = str((out_dir / f"data.{cif_id}_opt.lammps-data").absolute())

        with open(in_file, "r") as f:
            existing_content = f.read()

        # Fix read_data path to absolute
        if "read_data" in existing_content:
            lines = existing_content.split("\n")
            new_lines = []
            for line in lines:
                if line.strip().startswith("read_data"):
                    data_filename = line.split()[1]
                    data_path = (data_dir / data_filename).absolute()
                    new_lines.append(f"read_data       {data_path}")
                else:
                    new_lines.append(line)
            existing_content = "\n".join(new_lines)

        # Append optimization commands
        with open(in_file, "w") as f:
            f.write(existing_content)
            f.write(LAMMPS_MINIMIZE_EXE.format(cif_id, out_file))

        created_files.append(in_file)

    log_step("Optimize-Input", f"Created {len(created_files)} input files")
    return created_files


def run_lammps_optimization(input_file: Path, output_dir: Path) -> bool:
    """Run LAMMPS optimization on a single input file using nohup."""
    import subprocess

    log_dir = output_dir / "log"
    log_dir.mkdir(exist_ok=True, parents=True)

    filename = input_file.name
    cif_id = filename.replace("in.", "")

    out_lammps_data = output_dir / "opt_lammps_data" / f"data.{cif_id}_opt.lammps-data"
    out_lammps_data.parent.mkdir(exist_ok=True, parents=True)

    log_file = log_dir / f"out_{cif_id}.lammps"
    done_file = log_dir / f"in.{cif_id}.DONE.txt"
    error_file = log_dir / f"in.{cif_id}.ERROR.txt"

    if done_file.exists() and out_lammps_data.exists():
        log_info(f"Skipping {input_file.name} - already optimized")
        return True

    try:
        # FIX 2026-04-09: Han's original code hardcoded "lmp_mpi" in the cmd string,
        # ignoring the LAMMPS_EXECUTABLE env var defined at line 34. This patch
        # uses the env var so a custom binary path (e.g. Windows lmp.exe) is honored.
        # Also: nohup is unreliable on Windows Git Bash; use direct subprocess.run
        # in foreground for portability. For Linux/macOS production runs the
        # original nohup background pattern still works through the shell=True path.
        import platform
        if platform.system() == "Windows":
            # Windows: run lmp.exe synchronously (no nohup)
            cmd_list = [
                LAMMPS_EXECUTABLE,
                "-in", str(input_file.absolute()),
                "-log", str(log_file.absolute()),
            ]
            log_info(f"Running LAMMPS (sync, Windows): {LAMMPS_EXECUTABLE}")
            result = subprocess.run(
                cmd_list,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=900,  # 15 min cap for the smoke test
            )
            if result.returncode == 0:
                done_file.write_text("DONE\n")
                log_info(f"LAMMPS optimization done for {input_file.name}")
            else:
                log_error(
                    f"LAMMPS exited with {result.returncode} for {input_file.name}: "
                    f"{result.stderr[:500] if result.stderr else result.stdout[-500:]}"
                )
                with open(error_file, "w") as f:
                    f.write(result.stderr or result.stdout or "(no output)")
                return False
        else:
            # Linux/macOS: original nohup background pattern, with env var fix
            cmd = (
                f"nohup {LAMMPS_EXECUTABLE} -in {input_file.absolute()} "
                f"> {log_file.absolute()} 2>&1 "
                f"&& echo 'DONE' > {done_file.absolute()} &"
            )
            subprocess.Popen(cmd, shell=True, cwd=str(output_dir))
            log_info(f"Started LAMMPS optimization for {input_file.name} (background)")

        return True

    except Exception as e:
        log_error(f"Error running LAMMPS for {input_file.name}: {e}")
        with open(error_file, "w") as f:
            f.write(str(e))
        return False


def check_optimization_complete(output_dir: Path) -> bool:
    in_files_dir = output_dir / "in_files"
    in_files = list(in_files_dir.glob("in.*"))
    log_dir = output_dir / "log"

    for in_file in in_files:
        done_file = log_dir / f"{in_file.stem}.DONE.txt"
        if not done_file.exists():
            return False
    return True


def run_all_optimizations(data_dir: Path, output_dir: Path) -> None:
    """Run LAMMPS optimization on all input files in background."""
    in_files = list(data_dir.glob("in.*"))
    if not in_files:
        log_error(f"No in files found in {data_dir}")
        return

    log_step(
        "LAMMPS-Optimize",
        f"Creating optimization input files for {len(in_files)} files...",
    )

    created_inputs = make_optimization_input(data_dir, output_dir)

    if not created_inputs:
        log_error("No optimization input files created")
        return

    log_step(
        "LAMMPS-Optimize",
        f"Starting optimization on {len(created_inputs)} files (background)...",
    )

    for input_file in created_inputs:
        run_lammps_optimization(input_file, output_dir)

    log_step("LAMMPS-Optimize", f"Started {len(created_inputs)} jobs in background")


def wait_for_optimizations(output_dir: Path, check_interval: int = 10) -> None:
    import time

    data_dir = output_dir / "lammps_data"
    log_dir = output_dir / "log"

    if not data_dir.exists():
        return

    in_files = list(data_dir.glob("in.*"))
    if not in_files:
        return

    while True:
        all_done = True
        for in_file in in_files:
            done_file_name = in_file.name + ".DONE.txt"
            done_file = log_dir / done_file_name
            if not done_file.exists():
                all_done = False
                break

        if all_done:
            log_info("All LAMMPS optimizations completed!")
            return

        log_info(f"Waiting for {check_interval}s...")
        time.sleep(check_interval)


# =============================================================================
# Converter Step (LAMMPS data to CIF)
# =============================================================================
def convert_lammps_to_cif(input_file: Path, output_file: Path) -> bool:
    """
    Convert LAMMPS data file to CIF using converter.

    Args:
        input_file: Path to LAMMPS data file
        output_file: Path to output CIF file

    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess

        cmd = [
            sys.executable,
            CONVERTER_SCRIPT,
            "-i",
            str(input_file.absolute()),
            "-o",
            str(output_file.absolute()),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        return result.returncode == 0

    except Exception as e:
        log_error(f"Error converting {input_file.name}: {e}")
        return False


def convert_all_lammps_data(data_dir: Path, output_dir: Path) -> None:
    """
    Convert all LAMMPS data files to CIF format.

    Args:
        data_dir: Directory containing LAMMPS data files
        output_dir: Directory to save CIF files
    """
    lammps_files = list(data_dir.glob("*_opt.lammps-data"))
    if not lammps_files:
        log_error(f"No _opt.lammps-data files found in {data_dir}")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    log_step("Converter", f"Converting {len(lammps_files)} LAMMPS data files to CIF...")

    success_count = 0
    for lammps_file in lammps_files:
        name = lammps_file.stem.replace("data.", "").replace("_opt", "")
        cif_file = output_dir / f"{name}.cif"
        if convert_lammps_to_cif(lammps_file, cif_file):
            success_count += 1

    log_step("Converter", f"Completed {success_count}/{len(lammps_files)} files")


# =============================================================================
# Main Pipeline
# =============================================================================
def run_pipeline(
    cif_dir: Path,
    output_dir: Path,
    run_interface: bool = True,
    run_optimize: bool = True,
    run_convert: bool = True,
    lammps_interface_cmd: str = "lammps-interface",
) -> None:
    """
    Run the full optimization pipeline.

    Args:
        cif_dir: Input CIF directory
        output_dir: Output directory
        run_interface: Whether to run lammps-interface step
        run_optimize: Whether to run LAMMPS optimization step
        run_convert: Whether to convert results back to CIF
        lammps_interface_cmd: Command for lammps-interface
    """
    log_info(f"Starting optimization pipeline")
    log_info(f"Input CIF directory: {cif_dir}")
    log_info(f"Output directory: {output_dir}")

    # Step 1: Generate LAMMPS data files from CIF (with lammps-interface --minimize)
    if run_interface:
        data_dir = output_dir / "lammps_data"
        data_dir.mkdir(exist_ok=True, parents=True)
        make_lammps_interface_jobs(cif_dir, data_dir, lammps_interface_cmd)

    # Step 2: Run LAMMPS optimizations using input files from lammps-interface
    if run_optimize:
        data_dir = output_dir / "lammps_data"
        run_all_optimizations(data_dir, output_dir)

        if run_convert:
            log_info("Waiting for LAMMPS optimizations to complete...")
            wait_for_optimizations(output_dir, check_interval=10)

    # Step 3: Convert optimized structures back to CIF
    if run_convert:
        opt_data_dir = output_dir / "opt_lammps_data"
        if not opt_data_dir.exists():
            log_error(f"Optimized data directory not found: {opt_data_dir}")
            log_info("Run optimization step first: --no-interface --no-convert")
        else:
            convert_all_lammps_data(opt_data_dir, output_dir / "optimized_cifs")

    log_info("Pipeline completed!")


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LAMMPS Optimization Pipeline for MOF CIF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on default CIF directory
  python core/opt/optimize.py

  # Run on custom CIF directory
  python core/opt/optimize.py --cif-dir /path/to/cifs

  # Run only conversion step
  python core/opt/optimize.py --convert-only --cif-dir /path/to/cifs

  # Run interface and optimization, skip conversion
  python core/opt/optimize.py --no-convert --cif-dir /path/to/cifs
""",
    )

    parser.add_argument(
        "--cif-dir",
        type=str,
        default=DEFAULT_CIF_DIR,
        help=f"Input CIF directory (default: {DEFAULT_CIF_DIR})",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--lammps-interface-cmd",
        type=str,
        default="lammps-interface",
        help="Command for lammps-interface (default: lammps-interface)",
    )

    # Pipeline step flags
    parser.add_argument(
        "--interface-only", action="store_true", help="Run only lammps-interface step"
    )

    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Run only LAMMPS optimization step (assumes data files exist)",
    )

    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Run only conversion step (assumes lammps-data files exist)",
    )

    # Convenience flags
    parser.add_argument(
        "--no-interface", action="store_true", help="Skip lammps-interface step"
    )

    parser.add_argument(
        "--no-optimize", action="store_true", help="Skip LAMMPS optimization step"
    )

    parser.add_argument(
        "--no-convert", action="store_true", help="Skip conversion step"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Convert string paths to Path objects
    cif_dir = Path(args.cif_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Validate input directory
    if not cif_dir.exists():
        log_error(f"CIF directory does not exist: {cif_dir}")
        sys.exit(1)

    # Determine which steps to run
    run_interface = (
        not args.no_interface and not args.optimize_only and not args.convert_only
    )
    run_optimize = (
        not args.no_optimize and not args.interface_only and not args.convert_only
    )
    run_convert = (
        not args.no_convert and not args.interface_only and not args.optimize_only
    )

    # Handle exclusive options
    if args.interface_only:
        run_interface = True
        run_optimize = False
        run_convert = False
    elif args.optimize_only:
        run_interface = False
        run_optimize = True
        run_convert = False
    elif args.convert_only:
        run_interface = False
        run_optimize = False
        run_convert = True

    # Run the pipeline
    run_pipeline(
        cif_dir=cif_dir,
        output_dir=output_dir,
        run_interface=run_interface,
        run_optimize=run_optimize,
        run_convert=run_convert,
        lammps_interface_cmd=args.lammps_interface_cmd,
    )


if __name__ == "__main__":
    main()
