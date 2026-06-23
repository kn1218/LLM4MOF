"""
Run Zeo++ on a single CIF file and return geometry descriptors.

Computes: surface area (sa, m²/cm³), unit cell volume (cv, Å³),
density (g/cm³), void fraction (vf, cm³/cm³),
included sphere diameter (di, Å), free sphere diameter (df, Å),
included sphere along free path (dif, Å).

Note: -ha (high accuracy) flag is intentionally omitted for speed.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional


def run_zeo(
    cif_path: str,
    zeopp_bin: str,
    timeout: int = 120,
) -> Optional[dict]:
    """
    Run Zeo++ on a CIF file and parse geometry descriptors.

    Runs three Zeo++ calculations (no -ha flag):
      -sa  1.2 1.2 5000   → surface area (m²/cm³), density, cell volume
      -vol 1.2 1.2 50000  → void fraction (accessible volume)
      -res                 → pore size (Di, Df, Dif)

    Args:
        cif_path: Absolute path to input CIF file.
        zeopp_bin: Path to Zeo++ 'network' binary.
        timeout: Per-subprocess timeout in seconds.

    Returns:
        dict with keys: sa (m²/cm³), cv (Å³), density (g/cm³),
            vf (cm³/cm³), di (Å), df (Å), dif (Å)
        Returns None if any critical descriptor (di, df) is missing.
    """
    t0 = time.time()
    cif_path = os.path.abspath(cif_path)
    cif_dir = os.path.dirname(cif_path)
    cif_stem = Path(cif_path).stem

    sa_out = os.path.join(cif_dir, f"{cif_stem}.sa")
    vol_out = os.path.join(cif_dir, f"{cif_stem}.vol")
    res_out = os.path.join(cif_dir, f"{cif_stem}.res")

    try:
        subprocess.run(
            [zeopp_bin, "-sa", "1.2", "1.2", "5000", cif_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
        subprocess.run(
            [zeopp_bin, "-vol", "1.2", "1.2", "50000", cif_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
        subprocess.run(
            [zeopp_bin, "-res", cif_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"   [Zeo++] TIMEOUT for {cif_stem}")
        return None
    except Exception as e:
        print(f"   [Zeo++] ERROR for {cif_stem}: {e}")
        return None

    result: dict = {}

    # Parse .sa → sa (m²/cm³), cv (Å³), density (g/cm³)
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

    # Parse .vol → vf (accessible volume fraction)
    if os.path.isfile(vol_out):
        with open(vol_out) as f:
            for line in f:
                if "AV_Volume_fraction:" in line:
                    try:
                        result["vf"] = float(line.split("AV_Volume_fraction:")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
                    break

    # Parse .res → di, df, dif (Å)
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

    # Clean up temporary output files
    for tmp in [sa_out, vol_out, res_out]:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    # Require at minimum pore size descriptors
    if "di" not in result or "df" not in result:
        print(f"   [Zeo++] FAILED (no pore size) for {cif_stem} ({time.time()-t0:.1f}s)")
        return None

    elapsed = time.time() - t0
    print(
        f"   [Zeo++] {cif_stem}: di={result['di']:.2f} df={result['df']:.2f} "
        f"vf={result.get('vf', float('nan')):.3f} "
        f"density={result.get('density', float('nan')):.3f} ({elapsed:.1f}s)"
    )
    return result
