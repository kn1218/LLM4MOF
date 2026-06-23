"""
Zeo++ installation utilities.

Downloads, compiles, and manages the Zeo++ network binary.
Binary is stored in core/simulation/zeo/bin/network by default.
Can be transferred to HPC via scp (like forcefield files).

Zeo++ source: https://github.com/mharanczyk/zeoplusplus
"""

import os
import shutil
import subprocess
from typing import List, Optional

ZEO_GITHUB_URL = "https://github.com/mharanczyk/zeoplusplus/archive/refs/heads/master.tar.gz"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN_DIR = os.path.join(_THIS_DIR, "bin")
DEFAULT_BIN_PATH = os.path.join(DEFAULT_BIN_DIR, "network")


def find_zeopp_bin(extra_dirs: Optional[List[str]] = None) -> Optional[str]:
    """
    Find an existing Zeo++ binary.

    Search order:
      1. Default install location (core/simulation/zeo/bin/network)
      2. System PATH
      3. extra_dirs
      4. Common system paths

    Args:
        extra_dirs: Additional directories to search.

    Returns:
        Absolute path to binary, or None if not found.
    """
    # 1. Default local install
    if os.path.isfile(DEFAULT_BIN_PATH) and os.access(DEFAULT_BIN_PATH, os.X_OK):
        return DEFAULT_BIN_PATH

    # 2. System PATH
    which = shutil.which("network")
    if which:
        return which

    # 3. extra_dirs
    if extra_dirs:
        for d in extra_dirs:
            p = os.path.join(d, "network")
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p

    # 4. Common system paths
    common = [
        "/usr/local/bin/network",
        "/usr/bin/network",
        os.path.expanduser("~/zeo++-0.3/network"),
        os.path.expanduser("~/zeo++-0.3/bin/network"),
        os.path.expanduser("~/zeo++/network"),
        os.path.expanduser("~/zeo++/bin/network"),
    ]
    for p in common:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    return None


def install_zeopp(target_dir: Optional[str] = None, force: bool = False) -> str:
    """
    Download, compile, and install Zeo++ binary from GitHub source.

    Args:
        target_dir: Directory where the binary will be placed.
                    Defaults to core/simulation/zeo/bin/.
        force: Re-compile even if binary already exists.

    Returns:
        Path to the installed binary.

    Raises:
        RuntimeError: If download or compilation fails.
    """
    if target_dir is None:
        target_dir = DEFAULT_BIN_DIR
    os.makedirs(target_dir, exist_ok=True)

    bin_path = os.path.join(target_dir, "network")
    if os.path.isfile(bin_path) and not force:
        print(f"[Zeo++] Binary already exists: {bin_path}")
        return bin_path

    build_dir = os.path.join(target_dir, "_build")
    os.makedirs(build_dir, exist_ok=True)

    src_tar = os.path.join(build_dir, "zeo.tar.gz")
    src_dir = os.path.join(build_dir, "zeoplusplus-master")

    if not os.path.isdir(src_dir):
        print("[Zeo++] Downloading source from GitHub...")
        result = subprocess.run(
            ["wget", "-q", ZEO_GITHUB_URL, "-O", src_tar],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download Zeo++: {result.stderr}")
        subprocess.run(
            ["tar", "-xzf", src_tar, "-C", build_dir],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    voro_dir = os.path.join(src_dir, "voro++")
    if os.path.isdir(voro_dir):
        print("[Zeo++] Compiling voro++...")
        subprocess.run(
            ["make", "-C", voro_dir, "-j4"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )

    print("[Zeo++] Compiling Zeo++...")
    subprocess.run(
        ["make", "-C", src_dir, "clean"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    result = subprocess.run(
        ["make", "-C", src_dir, "-j4"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to compile Zeo++:\n{result.stderr[-1000:]}")

    src_bin = os.path.join(src_dir, "network")
    if not os.path.isfile(src_bin):
        raise RuntimeError("Zeo++ 'network' binary not found after compilation.")

    shutil.copy(src_bin, bin_path)
    os.chmod(bin_path, 0o755)
    print(f"[Zeo++] Binary installed: {bin_path}")
    return bin_path


def ensure_zeopp(target_dir: Optional[str] = None, extra_dirs: Optional[List[str]] = None) -> str:
    """
    Find an existing Zeo++ binary or compile one from source.

    Args:
        target_dir: Where to install if not found.
        extra_dirs: Extra directories to search first.

    Returns:
        Path to Zeo++ binary.
    """
    bin_path = find_zeopp_bin(extra_dirs)
    if bin_path:
        print(f"[Zeo++] Found existing binary: {bin_path}")
        return bin_path
    return install_zeopp(target_dir)


if __name__ == "__main__":
    path = ensure_zeopp()
    print(f"Zeo++ binary: {path}")
