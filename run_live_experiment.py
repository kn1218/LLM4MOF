# =============================================================================
# LLM4MOF Autonomous System v3 - Live Simulation Entry Point
# =============================================================================
# run_live_experiment.py
# Mirrors run_experiment.py but replaces the markscheme/sensitivity-analyzer
# path with real PORMAKE → LAMMPS → RASPA3 simulations.
#
# Usage:
#   conda activate <your-env>
#   python run_live_experiment.py
#   python run_live_experiment.py --resume <runid>
#   python run_live_experiment.py --smoke   (1/beam, 200 cycles, quick validation)
# =============================================================================

import os
import sys
import datetime
import json
import argparse
import subprocess
import time
import re

# Fix Unicode encoding on Windows (Korean locale cp949 can't handle Å, ², ³)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # Prevent OpenMP duplicate lib crash (torch + numpy on Windows)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torch must be imported before any module that transitively loads it
# (filter_candidate → torch) to avoid Windows DLL search-order issues.
try:
    import torch  # noqa: F401
except OSError:
    pass  # torch DLL may be unavailable; mof2zeo loaded lazily when needed

from config import DEFAULT_INQUIRY, EXPERIMENTS_DIR, ACTIVE_MODEL, validate_api_keys
import config

from core.agent1_handler import Agent1Handler
from core.agent2_handler import Agent2Handler
from core.matchmaker import Matchmaker
from core.feedback_generator import FeedbackGenerator
from core.memory_manager import MemoryManager, ExperimentLogger
from core.live_runner import run_live_iteration, prepare_beam_pools, SimCache, compute_geometry_match_score
from core.feedback_live_adapter import live_results_to_filter_sets
from core.hpc.prepare_batch import prepare_manifest, prepare_r1_manifest, prepare_r2_manifest
from core.hpc.collect_results import collect_results


def print_banner(smoke: bool = False) -> None:
    mode_label = "SMOKE TEST" if smoke else "LIVE SIMULATION"
    print("\n" + "=" * 60)
    print(f"   LLM4MOF AUTONOMOUS MOF DESIGNER v3 — {mode_label}")
    print(f"   Model: {ACTIVE_MODEL}")
    print(f"   Beams: {config.LIVE_SIM_N_BEAMS} × "
          f"{config.LIVE_SIM_N_PER_BEAM} successes/beam")
    print(f"   RASPA cycles: {config.LIVE_SIM_RASPA_CYCLES} prod + "
          f"{config.LIVE_SIM_RASPA_INIT_CYCLES} init")
    print(f"   LAMMPS: {'ON' if not config.LIVE_SIM_SKIP_LAMMPS else 'SKIP'}")
    print(f"   Max iterations: {config.LIVE_SIM_MAX_ITERATIONS}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM4MOF Live Simulation Experiment"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume a previous run by its experiment directory name",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 1/beam, 200 RASPA cycles, 1 iteration",
    )
    parser.add_argument(
        "--inquiry", type=str, default=None,
        help="Direct inquiry string (skip interactive prompt)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override max iterations",
    )
    parser.add_argument(
        "--hpc", action="store_true",
        help="HPC mode: single-process loop with SSH polling (production)",
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="HPC manual mode: run LLM agents + matchmaker, generate manifest, then exit",
    )
    parser.add_argument(
        "--collect", action="store_true",
        help="HPC manual mode: parse downloaded HPC results, generate feedback, continue",
    )
    parser.add_argument(
        "--adsorbate", type=str, default="h2",
        choices=["h2", "ch4", "co2", "xekr"],
        help="Adsorbate type (default: h2). Sets T/P/forcefield/ChargeMethod from LIVE_SIM_ADSORBATE_CONFIGS.",
    )
    parser.add_argument(
        "--xe-molfrac", type=float, default=0.20,
        help="Xe mole fraction for xekr mixture (default: 0.20).",
    )
    parser.add_argument(
        "--pressure", type=float, default=None,
        help="Override RASPA pressure in bar (default: adsorbate config value). E.g., --pressure 5.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override RASPA temperature in K (default: adsorbate config value). E.g., --temperature 87.",
    )
    parser.add_argument(
        "--no-zeo", dest="zeo", action="store_false",
        help="Disable Zeo++ geometry computation (enabled by default).",
    )
    parser.add_argument(
        "--pormake-unit", type=str, default=None, dest="pormake_unit",
        choices=["volumetric", "molkg", "gperL"],
        help="Force PorMake unit variant (bypasses keyword detection). "
             "Used to avoid leaking 'gravimetric'/'mol/kg' in the inquiry.",
    )
    parser.add_argument(
        "--packed", action="store_true",
        help="Use the packed submit variant (multiple jobs per submission).",
    )
    parser.add_argument(
        "--server", type=str, default=None,
        help="Override HPC server hostname (default: config.HPC_HOST).",
    )
    parser.add_argument(
        "--node-prop", type=str, default=None, dest="node_prop",
        help="Override PBS node property (default: config.HPC_NODE_PROPERTY).",
    )
    parser.add_argument(
        "--job-prefix", type=str, default="llm4mof", dest="job_prefix",
        help="PBS job name prefix for queue isolation when running parallel experiments. "
             "Default 'llm4mof' preserves existing behavior. E.g., --job-prefix xekr.",
    )
    parser.set_defaults(zeo=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HPC SSH helpers
# ---------------------------------------------------------------------------

def _ssh_run(cmd: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    """
    Run a command on HPC via SSH with retry logic.

    Uses config.HPC_HOST and retries on connection failure with
    exponential backoff (config.HPC_SSH_RETRY_DELAYS).
    """
    # Prepend PBS bin to PATH for non-interactive SSH sessions
    full_cmd = f"export PATH=/usr/local/pbs/bin:$PATH; {cmd}"
    ssh_cmd = ["ssh", config.HPC_HOST, full_cmd]
    delays = config.HPC_SSH_RETRY_DELAYS

    for attempt in range(config.HPC_SSH_RETRIES):
        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace",
            )
            if result.returncode == 0 or not check:
                return result
            # Non-zero exit but not a connection error — return as-is
            if result.returncode != 255:
                return result
            # SSH error (rc=255) — retry
            print(f"   [SSH] Connection error (attempt {attempt + 1}/"
                  f"{config.HPC_SSH_RETRIES}): {result.stderr.strip()[:100]}")
        except subprocess.TimeoutExpired:
            print(f"   [SSH] Timeout (attempt {attempt + 1}/{config.HPC_SSH_RETRIES})")

        if attempt < config.HPC_SSH_RETRIES - 1:
            delay = delays[min(attempt, len(delays) - 1)]
            print(f"   [SSH] Retrying in {delay}s...")
            time.sleep(delay)

    raise ConnectionError(
        f"SSH to {config.HPC_HOST} failed after {config.HPC_SSH_RETRIES} attempts"
    )


def _scp_upload(local_path: str, remote_path: str) -> None:
    """Upload a file to HPC via scp, converting CRLF→LF for shell scripts."""
    # Convert Windows line endings for scripts/text files destined for Linux
    if local_path.endswith((".sh", ".py")):
        import tempfile
        with open(local_path, "rb") as f:
            content = f.read()
        if b"\r\n" in content:
            converted = content.replace(b"\r\n", b"\n")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.basename(local_path))
            tmp.write(converted)
            tmp.close()
            try:
                subprocess.run(
                    ["scp", tmp.name, f"{config.HPC_HOST}:{remote_path}"],
                    check=True, capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
                )
            finally:
                os.unlink(tmp.name)
            return
    subprocess.run(
        ["scp", local_path, f"{config.HPC_HOST}:{remote_path}"],
        check=True, capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
    )


def _scp_download(remote_path: str, local_path: str) -> None:
    """Download a file from HPC via scp."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    subprocess.run(
        ["scp", f"{config.HPC_HOST}:{remote_path}", local_path],
        check=True, capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
    )


def _scp_upload_dir(local_dir: str, remote_dir: str) -> None:
    """Upload a directory to HPC via scp -r."""
    subprocess.run(
        ["scp", "-r", local_dir, f"{config.HPC_HOST}:{remote_dir}"],
        check=True, capture_output=True, text=True, timeout=300,
    )


# _compute_geometry_match_score is defined in core/live_runner.py and imported above
# as compute_geometry_match_score.  Keep this alias so HPC-path callsites work unchanged.
_compute_geometry_match_score = compute_geometry_match_score

_PACMAN_BUFFER_SIZE = 15      # candidates per beam through local GPU charge stage
_PACMAN_TIMEOUT_S = 300       # per-structure timeout (seconds)


def run_local_charge_stage(
    z_stage2_jobs: list,
    af_total_stage2_jobs: dict,
    iter_dir: str,
    hpc_iter_dir: str,
) -> tuple:
    """Assign DDEC6 framework charges on the local GPU (PACMAN-charge) between R1 and R2.

    Downloads CIFs from HPC, runs PACMAN-charge locally with a per-structure timeout,
    uploads ``{name}_pacman.cif`` back to the same HPC directory, and returns updated
    job lists with ``cif_path`` pointing to the pre-charged CIF.  Jobs where PACMAN
    fails or times out are excluded from the returned lists.

    Args:
        z_stage2_jobs: Z-beam R2 candidates (already geometry-ranked, up to _PACMAN_BUFFER_SIZE).
        af_total_stage2_jobs: {beam_id: [job_dict, ...]} for A/F/total beams.
        iter_dir: Local experiment iteration directory.
        hpc_iter_dir: HPC-side experiment iteration directory.

    Returns:
        (z_jobs_charged, af_total_jobs_charged) — same structure as inputs but filtered
        to PACMAN successes only, with ``cif_path`` updated to the ``_pacman.cif`` path
        on the HPC side.
    """
    import random
    import shutil

    charge_tmp = os.path.join(iter_dir, "charge_tmp")
    os.makedirs(charge_tmp, exist_ok=True)

    try:
        from PACMANCharge import pmcharge
    except ImportError:
        print("[CHARGE] PACMAN-charge not installed locally; skipping charge stage")
        return z_stage2_jobs, af_total_stage2_jobs

    def _run_pacman(local_cif: str) -> str | None:
        """Run PACMAN on *local_cif* with timeout; return path to _pacman.cif or None."""
        import threading

        pacman_out = local_cif.replace(".cif", "_pacman.cif")
        result = {"success": False, "error": ""}

        def _worker():
            try:
                pmcharge.predict(
                    cif_file=local_cif, charge_type="DDEC6", digits=6,
                    atom_type=True, neutral=True, keep_connect=True,
                )
                result["success"] = os.path.isfile(pacman_out)
            except Exception as exc:
                result["error"] = str(exc)[:200]

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=_PACMAN_TIMEOUT_S)
        if t.is_alive():
            print(f"   [CHARGE] Timeout ({_PACMAN_TIMEOUT_S}s): {os.path.basename(local_cif)}")
            return None
        if not result["success"]:
            print(f"   [CHARGE] Failed ({result['error']}): {os.path.basename(local_cif)}")
            # Clean up partial output if present
            if os.path.isfile(pacman_out):
                os.remove(pacman_out)
            return None
        return pacman_out

    def _charge_jobs(jobs: list) -> list:
        """Run PACMAN on each job; return list of successfully charged jobs (cif_path updated)."""
        charged = []
        for job in jobs:
            hpc_cif = job.get("cif_path")
            if not hpc_cif:
                print(f"   [CHARGE] No cif_path for {job.get('filename')}; skipping")
                continue
            name = os.path.basename(hpc_cif)
            local_cif = os.path.join(charge_tmp, name)
            hpc_cif_dir = os.path.dirname(hpc_cif)

            # Download CIF from HPC
            try:
                _scp_download(hpc_cif, local_cif)
            except Exception as exc:
                print(f"   [CHARGE] SCP download failed for {name}: {exc}")
                continue

            # Run PACMAN locally
            print(f"   [CHARGE] {job.get('filename')} ...", end=" ", flush=True)
            t0 = time.time()
            pacman_cif = _run_pacman(local_cif)
            elapsed = time.time() - t0

            if pacman_cif is None:
                print(f"skipped ({elapsed:.1f}s)")
                continue

            print(f"ok ({elapsed:.1f}s)")

            # Upload _pacman.cif back to HPC (same directory as original CIF)
            hpc_pacman_cif = os.path.join(hpc_cif_dir, os.path.basename(pacman_cif))
            try:
                _scp_upload(pacman_cif, hpc_pacman_cif)
            except Exception as exc:
                print(f"   [CHARGE] SCP upload failed for {os.path.basename(pacman_cif)}: {exc}")
                continue

            updated_job = dict(job)
            updated_job["cif_path"] = hpc_pacman_cif
            charged.append(updated_job)

        return charged

    # --- Z beam ---
    print(f"\n[CHARGE] Local GPU charge stage: {len(z_stage2_jobs)} Z + "
          f"{sum(len(v) for v in af_total_stage2_jobs.values())} A/F/total")
    z_charged = _charge_jobs(z_stage2_jobs)
    print(f"[CHARGE] Z: {len(z_charged)}/{len(z_stage2_jobs)} charged")

    # --- A/F/total beams ---
    af_charged: dict = {}
    for beam_id, jobs in af_total_stage2_jobs.items():
        charged = _charge_jobs(jobs)
        af_charged[beam_id] = charged
        print(f"[CHARGE] {beam_id}: {len(charged)}/{len(jobs)} charged")

    # Clean up local CIFs
    shutil.rmtree(charge_tmp, ignore_errors=True)

    return z_charged, af_charged


def _apply_zeo_postprocess(live_results: "LiveResults") -> None:
    """
    Post-process LiveResults after HPC collection when Zeo++ was used.

    Removes zeo_fail entries from all beams so only structures with valid
    real_geometry are passed to feedback. Selection of top-8 is left to
    FeedbackGenerator.nlargest(8, 'target') uniformly across all beams.
    """
    for beam_id, beam_result in live_results.beams.items():
        zeo_failed = [r for r in beam_result.successes if r.status != "success"]
        kept = [r for r in beam_result.successes if r.status == "success"]

        n_removed = len(zeo_failed)
        beam_result.failures.extend(zeo_failed)
        live_results.n_failures += n_removed
        live_results.n_real_simulations -= n_removed
        beam_result.successes = kept

        if n_removed > 0:
            print(f"[ZeoPost] Beam {beam_id}: removed {n_removed} zeo_fail entries")


def _build_matchmaker_failure_feedback(
    requested_metals: list,
    requested_cn: list,
    requested_node_abs: dict,
) -> str:
    """Build informative matchmaker failure feedback with available BB summary.

    Instead of a blanket 'Do NOT use X', tells Agent 1 exactly which
    (metal, CN) combos are available so it can make a minimal adjustment.
    """
    import json as _json
    from collections import defaultdict

    bb_path = config.BB_DICTIONARY_PATH
    bb_data = _json.load(open(bb_path, encoding="utf-8"))
    nodes = [b for b in bb_data if b.get("Type") == "Node"]

    # Build (metal, CN) -> count map
    metal_cn_map = defaultdict(int)
    for n in nodes:
        for m in n.get("metals", []):
            metal_cn_map[(m, n["connectivity"])] += 1

    # --- Diagnose: which requested metals have which CNs? ---
    metals_str = ", ".join(requested_metals) if requested_metals else "requested metals"
    cn_str = str(requested_cn) if requested_cn else "unknown"

    # Per requested metal, show what CNs are actually available
    available_lines = []
    for m in requested_metals:
        combos = sorted([(cn, cnt) for (mm, cn), cnt in metal_cn_map.items() if mm == m], key=lambda x: x[0])
        if combos:
            combo_str = ", ".join(f"CN={cn} ({cnt} nodes)" for cn, cnt in combos)
            available_lines.append(f"  {m}: {combo_str}")
        else:
            available_lines.append(f"  {m}: not available in database")


    # --- Check if node abstract features caused the failure ---
    abs_note = ""
    if requested_node_abs:
        active_abs = {k: v for k, v in requested_node_abs.items() if v is not None}
        if active_abs:
            # Count how many requested-metal nodes match the abstract features
            def _node_matches_abs(node, abs_feats):
                nf = node.get("abstract_features", {})
                for k, v in abs_feats.items():
                    if v is not None and nf.get(k) != v:
                        return False
                return True

            n_with_metal = [n for n in nodes
                            if any(m in n.get("metals", []) for m in requested_metals)
                            and n["connectivity"] in (requested_cn if isinstance(requested_cn, list) else [requested_cn])]
            n_with_abs = [n for n in n_with_metal if _node_matches_abs(n, active_abs)]
            if len(n_with_metal) > 0 and len(n_with_abs) == 0:
                abs_note = (
                    f"\nNote: {len(n_with_metal)} nodes match your metals+CN, but 0 match "
                    f"the additional abstract features {active_abs}.\n"
                    f"Consider relaxing or removing abstract feature constraints on nodes.\n"
                )

    feedback = (
        f"[MATCHMAKER FAILURE — No simulations ran this iteration]\n\n"
        f"Your hypothesis could not be matched to any building blocks in the PorMake database.\n\n"
        f"  Beam Z (Full Hypothesis): 0 candidates\n"
        f"  Beam A (Chemistry Only):  0 candidates\n"
        f"  Beam F (Metal Only):      0 candidates\n\n"
        f"Requested: metals={metals_str}, connectivity={cn_str}\n\n"
        f"Available connectivity for your requested metals:\n"
        + "\n".join(available_lines) + "\n"
        + abs_note
        + f"\nAction: adjust ONLY the failing constraint (connectivity or abstract features).\n"
        f"Keep the rest of your hypothesis — especially chemistry and geometry insights\n"
        f"from previous successful iterations.\n"
    )
    return feedback


def _hpc_poll(
    hpc_iter_dir: str,
    results_remote_dir: str,
    n_jobs: int,
    beam_filenames: dict,
    job_prefix: str = "llm4mof",
) -> int:
    """
    Poll HPC via SSH until all .DONE files appear or per-beam sufficiency is met.

    Args:
        hpc_iter_dir: Remote path to iter_N directory (for queue check context)
        results_remote_dir: Remote path where .DONE files appear
        n_jobs: Total number of jobs submitted
        beam_filenames: {beam_id: [filename, ...]} for per-beam progress tracking

    Returns:
        Number of DONE files when polling terminates
    """
    from collections import defaultdict
    max_polls = int(config.HPC_POLL_MAX_HOURS * 3600 / config.HPC_POLL_INTERVAL)
    n_done = 0
    _stale_count = 0           # consecutive polls where queue_empty=True and n_done unchanged
    _STALE_THRESHOLD = 3       # trigger resubmit after this many stale polls
    _MAX_RESUBMIT = 2          # max resubmit attempts per polling session
    _resubmit_count = 0

    for poll in range(1, max_polls + 1):
        time.sleep(config.HPC_POLL_INTERVAL)

        done_result = _ssh_run(
            f"ls {results_remote_dir}/*.DONE 2>/dev/null | xargs -I{{}} basename {{}} .DONE",
            check=False,
        )
        done_set = set(done_result.stdout.strip().split("\n")) if done_result.stdout.strip() else set()
        prev_n_done = n_done
        n_done = len(done_set)

        beam_done: dict = defaultdict(int)
        for bid, fnames in beam_filenames.items():
            beam_done[bid] = sum(1 for f in fnames if f in done_set)

        min_beam_done = min(beam_done.values()) if beam_done else 0
        beam_summary = ", ".join(f"{b}:{beam_done[b]}" for b in sorted(beam_done))

        queue_empty = False
        q_result = _ssh_run(
            f"{config.HPC_STATUS_CMD} 2>/dev/null | grep {job_prefix} | wc -l",
            check=False,
        )
        try:
            queue_empty = int(q_result.stdout.strip()) == 0
        except ValueError:
            queue_empty = False

        # Stale detection: queue empty and no new DONE files
        if queue_empty and n_done == prev_n_done:
            _stale_count += 1
        else:
            _stale_count = 0

        elapsed_min = poll * config.HPC_POLL_INTERVAL / 60
        stale_note = f" [stale {_stale_count}/{_STALE_THRESHOLD}]" if queue_empty and _stale_count > 0 else ""
        print(f"   [HPC Poll #{poll}] {n_done}/{n_jobs} DONE [{beam_summary}] "
              f"queue_empty={queue_empty} ({elapsed_min:.0f} min elapsed){stale_note}", flush=True)

        if n_done >= n_jobs:
            print(f"[HPC] All {n_jobs} jobs complete!")
            break

        if queue_empty and min_beam_done >= config.LIVE_SIM_N_PER_BEAM:
            print(f"[HPC] Queue empty and each beam has ≥{config.LIVE_SIM_N_PER_BEAM} results "
                  f"({n_done}/{n_jobs} DONE). Proceeding.")
            break

        if queue_empty and _stale_count >= _STALE_THRESHOLD:
            # Find qsub scripts for jobs that never produced a .DONE file
            all_fnames = [f for fnames in beam_filenames.values() for f in fnames]
            missing = [f for f in all_fnames if f not in done_set]
            qsub_dir = f"{results_remote_dir}/qsub_scripts"

            if missing and _resubmit_count < _MAX_RESUBMIT:
                # Grep qsub scripts that reference any missing filename
                pattern = "|".join(re.escape(f) for f in missing)
                find_result = _ssh_run(
                    f"grep -rl -E '{pattern}' {qsub_dir}/ 2>/dev/null | sort -u",
                    check=False,
                )
                scripts = [s.strip() for s in find_result.stdout.strip().split("\n") if s.strip()]
                if scripts:
                    scripts_str = " ".join(scripts)
                    _resubmit_count += 1
                    print(f"[HPC] Stale detected: resubmitting {len(scripts)} qsub scripts "
                          f"for {len(missing)} missing jobs "
                          f"(attempt {_resubmit_count}/{_MAX_RESUBMIT})...", flush=True)
                    _ssh_run(f"cd {hpc_iter_dir} && {config.HPC_SUBMIT_CMD} {scripts_str}", check=False)
                    _stale_count = 0
                    continue

            print(f"[HPC WARNING] Queue empty and no new DONE files for {_STALE_THRESHOLD} consecutive polls "
                  f"({n_done}/{n_jobs} DONE, {len(missing)} missing). "
                  f"Jobs likely lost after {_resubmit_count} resubmit attempt(s). Proceeding.")
            break
    else:
        print(f"[HPC WARNING] Timeout after {config.HPC_POLL_MAX_HOURS}h "
              f"({n_done}/{n_jobs} complete). Proceeding with partial results.")

    return n_done


def _build_geometry_abort_feedback(stats: dict, constraints: dict) -> str:
    """Build human-readable geometry abort feedback for Agent 1."""
    gf = constraints.get("geometry_filter", {}) or {}
    prop_labels = {
        "di":      ("Di (largest included sphere)", "Å"),
        "df":      ("Df (largest free sphere)",     "Å"),
        "sa":      ("Surface area",                 "m²/g"),
        "vf":      ("Void fraction",                ""),
        "density": ("Density",                      "g/cm³"),
    }
    geo_lines = []
    for gkey, (label, unit) in prop_labels.items():
        if gkey not in stats:
            continue
        s = stats[gkey]
        tmin = s["target_min"]
        tmax = s["target_max"]
        if tmin is None and tmax is None:
            continue
        target_str = (
            f"{tmin}–{tmax}" if tmin is not None and tmax is not None
            else (f"≥{tmin}" if tmin is not None else f"≤{tmax}")
        )
        unit_str = f" {unit}" if unit else ""
        geo_lines.append(
            f"  {label}: target={target_str}{unit_str} | "
            f"actual mean={s['mean']}{unit_str}, range=[{s['min']}, {s['max']}]{unit_str}"
        )

    return (
        f"[GEOMETRY ABORT — No simulations ran this iteration]\n\n"
        f"None of the MOFs built in Phase 1 (PORMAKE → LAMMPS → Zeo++) passed "
        f"your geometry filter. RASPA was not run.\n\n"
        f"Comparison of your targets vs. what was actually built:\n"
        + "\n".join(geo_lines) + "\n\n"
        f"Required action for next iteration:\n"
        f"  - Relax your geometry constraints so they overlap with what PORMAKE can build.\n"
        f"  - OR change topology/node choice to target a different pore size regime.\n"
        f"  - The 'actual range' above shows what the current chemistry/topology combination produces.\n"
    )


def _hpc_simulate_two_stage(
    beam_pools: dict,
    iteration: int,
    experiment_id: str,
    experiment_dir: str,
    sim_cache,
    geometry_filter: dict,
    hpc_iter_dir: str,
    hpc_remote_scripts: str,
    zeo_flag: str,
    ads_flag: str = "",
    job_prefix: str = "llm4mof",
) -> "LiveResults":
    """
    Two-stage HPC pipeline (--zeo mode):

      Round 1: Z-100 + A/F/total-30each stage1_only (pormake+lammps+zeo) → results_r1/
      Filter Z: geometry_filter → top LIVE_SIM_Z_RASPA_TOP (with fallback)
      Filter A/F/total: top LIVE_SIM_N_PER_BEAM successes per beam
      Round 2: Z-top15 + A/F/total-top10 stage2_only (RASPA only) → results/

    Returns LiveResults parsed from Round 2 results.
    """
    from collections import defaultdict

    iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
    _ads_cfg = config.LIVE_SIM_ADSORBATE_CONFIGS.get(config.LIVE_SIM_ADSORBATE, {})
    _charge_method = _ads_cfg.get("charge_method", "None")
    _use_pacman = _charge_method == "Ewald"

    # ===========================================================
    # ROUND 1: Z-100 + A/F/total-30 stage1_only
    # ===========================================================
    af_total_pools = {k: v[:config.LIVE_SIM_AF_POOL_SIZE] for k, v in beam_pools.items() if k != "Z"}
    z_pool_actual = len(beam_pools.get("Z", []))
    print(f"\n[HPC Two-Stage] Round 1: {z_pool_actual} Z + "
          f"{sum(len(v) for v in af_total_pools.values())} A/F/total candidates "
          f"(pormake+lammps+zeo only)")

    z_pool = beam_pools.get("Z", [])[:config.LIVE_SIM_Z_POOL_SIZE]
    r1_manifest_path = prepare_r1_manifest(
        z_beam_pool=z_pool,
        iteration=iteration,
        experiment_id=experiment_id,
        iter_dir=iter_dir,
        af_total_beam_pools=af_total_pools,
    )
    with open(r1_manifest_path, "r") as f:
        r1_manifest_data = json.load(f)
    n_r1_jobs = r1_manifest_data["n_jobs"]

    if n_r1_jobs == 0:
        print("[HPC Two-Stage] No Z candidates — skipping two-stage, using standard pipeline")
        return _hpc_simulate_single(
            beam_pools=beam_pools, iteration=iteration, experiment_id=experiment_id,
            experiment_dir=experiment_dir, sim_cache=sim_cache,
            use_zeo=True, geometry_filter=geometry_filter,
            hpc_iter_dir=hpc_iter_dir, hpc_remote_scripts=hpc_remote_scripts, zeo_flag=zeo_flag,
            job_prefix=job_prefix,
        )

    _ssh_run(f"mkdir -p {hpc_iter_dir}/results_r1")
    _scp_upload(r1_manifest_path, f"{hpc_iter_dir}/batch_manifest_r1.json")

    print(f"[HPC] Submitting {n_r1_jobs} Round-1 jobs (stage1_only)...")
    r1_submit = _ssh_run(
        f"cd {hpc_iter_dir} && JOB_PREFIX={job_prefix} bash {hpc_remote_scripts}/{config.HPC_SUBMIT_SCRIPT} "
        f"batch_manifest_r1.json results_r1 {config.HPC_NODE_PROPERTY}{zeo_flag}{ads_flag}",
        timeout=600,
    )
    print(f"[HPC] R1 submit: {(r1_submit.stdout or '').strip()[-200:]}")
    if r1_submit.returncode != 0:
        raise RuntimeError(f"R1 job submission failed (rc={r1_submit.returncode})")

    r1_beam_filenames: dict = defaultdict(list)
    for job in r1_manifest_data["jobs"]:
        r1_beam_filenames[job.get("beam_id", "Z")].append(job["filename"])
    _hpc_poll(hpc_iter_dir, f"{hpc_iter_dir}/results_r1", n_r1_jobs, r1_beam_filenames, job_prefix)

    r1_agg = _ssh_run(
        f"cd {hpc_iter_dir} && python {hpc_remote_scripts}/aggregate_results.py "
        f"--manifest batch_manifest_r1.json --results-dir results_r1"
    )
    print(f"[HPC] R1 aggregate: {(r1_agg.stdout or '').strip()[-200:]}")

    local_r1_dir = os.path.join(iter_dir, "hpc_results_r1")
    os.makedirs(local_r1_dir, exist_ok=True)
    local_r1_path = os.path.join(local_r1_dir, "batch_results_r1.json")
    _scp_download(f"{hpc_iter_dir}/results_r1/batch_results.json", local_r1_path)

    # Parse R1 results
    with open(local_r1_path, "r") as f:
        r1_data = json.load(f)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpc"))
    from run_mof_sim import passes_geometry_filter

    all_r1 = r1_data.get("results", [])
    all_stage1_done = [r for r in all_r1 if r.get("status") == "stage1_done" and r.get("real_geometry")]
    z_stage1_done = [r for r in all_stage1_done if r.get("beam_id") == "Z"]
    n_total = len(all_r1)

    # --- Z beam: geometry filter ---
    has_geo_filter = bool(geometry_filter) and any(v is not None for v in geometry_filter.values())
    if has_geo_filter:
        z_passing = [r for r in z_stage1_done if passes_geometry_filter(r["real_geometry"], geometry_filter)]
        n_strict = len(z_passing)
        print(f"[HPC Two-Stage] Z geometry filter: {n_strict}/{len(z_stage1_done)} pass strict filter")

        if n_strict == 0:
            geo_keys = [
                ("di",      "target_Di_min",      "target_Di_max"),
                ("df",      "target_Df_min",      "target_Df_max"),
                ("sa",      "target_sa_min",      "target_sa_max"),
                ("vf",      "target_vf_min",      "target_vf_max"),
                ("density", "target_density_min", "target_density_max"),
            ]
            stats: dict = {}
            for gkey, min_key, max_key in geo_keys:
                vals = [r["real_geometry"][gkey] for r in z_stage1_done
                        if r.get("real_geometry") and gkey in r["real_geometry"]]
                if vals:
                    stats[gkey] = {
                        "mean": round(sum(vals) / len(vals), 3),
                        "min": round(min(vals), 3),
                        "max": round(max(vals), 3),
                        "target_min": geometry_filter.get(min_key),
                        "target_max": geometry_filter.get(max_key),
                    }
            from core.live_runner import LiveResults
            lr = LiveResults()
            lr.geometry_aborted = True
            lr.stage1_geometry_stats = stats
            print(f"[HPC Two-Stage] GEOMETRY ABORT: 0/{len(z_stage1_done)} Z pass geometry filter "
                  f"→ aborting iteration")
            return lr

        from core.live_runner import compute_geometry_match_score, geometry_fail_reason
        n_top = _PACMAN_BUFFER_SIZE if _use_pacman else config.LIVE_SIM_Z_RASPA_TOP
        if n_strict < n_top:
            not_passing = [r for r in z_stage1_done
                           if not passes_geometry_filter(r["real_geometry"], geometry_filter)]
            not_passing.sort(
                key=lambda r: compute_geometry_match_score(r["real_geometry"], geometry_filter),
                reverse=True,
            )
            n_fallback = min(n_top - n_strict, len(not_passing))
            z_top = z_passing[:n_top] + not_passing[:n_fallback]
            print(f"[HPC Two-Stage] Z: {n_strict} strict + {n_fallback} fallback → {len(z_top)} RASPA candidates")
        else:
            z_top = z_passing[:n_top]
            not_passing = []
            n_fallback = 0
            print(f"[HPC Two-Stage] Z top-{len(z_top)} selected for RASPA (no fallback needed)")

        # Metal-stratified selection: prevent one metal from monopolising all RASPA slots
        if config.is_stratified_sampling():
            from core.sampling_strata import stratified_select_dicts
            z_top = stratified_select_dicts(z_top, n_top)
            print(f"[HPC Two-Stage] Z: stratified → {len(z_top)} RASPA candidates")

        strict_filenames = {r["filename"] for r in z_passing[:n_top]}
        from core.live_runner import geometry_fail_reason
        z_stage2_jobs = [
            {
                "filename": r["filename"],
                "topology": r["topology"],
                "node": r["node"],
                "edge": r["edge"],
                "cif_path": r.get("cif_path"),
                "real_geometry": r.get("real_geometry"),
                "match_score": r.get("match_score", 0.0),
                "predicted_geometry": r.get("predicted_geometry"),
                "geo_filter_passed": r["filename"] in strict_filenames,
                "geo_filter_fail_reason": (
                    "" if r["filename"] in strict_filenames
                    else geometry_fail_reason(r.get("real_geometry", {}), geometry_filter)
                ),
            }
            for r in z_top
        ]
    else:
        # No geometry filter: take top-N Z successes (already in ranked order)
        z_top = z_stage1_done[:(_PACMAN_BUFFER_SIZE if _use_pacman else config.LIVE_SIM_Z_RASPA_TOP)]
        print(f"[HPC Two-Stage] Z: no geometry filter → top-{len(z_top)} successes for RASPA")
        z_stage2_jobs = [
            {
                "filename": r["filename"],
                "topology": r["topology"],
                "node": r["node"],
                "edge": r["edge"],
                "cif_path": r.get("cif_path"),
                "real_geometry": r.get("real_geometry"),
                "match_score": r.get("match_score", 0.0),
                "predicted_geometry": r.get("predicted_geometry"),
                "geo_filter_passed": True,
                "geo_filter_fail_reason": "",
            }
            for r in z_top
        ]

    # --- A/F/total: pick top N successes per beam from R1 ---
    n_per_beam = _PACMAN_BUFFER_SIZE if _use_pacman else config.LIVE_SIM_N_PER_BEAM
    af_total_stage2_jobs: dict = {}
    for beam_id in ("A", "F", "total"):
        beam_done = [r for r in all_stage1_done if r.get("beam_id") == beam_id]
        top_n = beam_done[:n_per_beam]
        af_total_stage2_jobs[beam_id] = [
            {
                "filename": r["filename"],
                "topology": r["topology"],
                "node": r["node"],
                "edge": r["edge"],
                "cif_path": r.get("cif_path"),
                "real_geometry": r.get("real_geometry"),
                "match_score": r.get("match_score", 0.0),
                "predicted_geometry": r.get("predicted_geometry"),
            }
            for r in top_n
        ]
        print(f"[HPC Two-Stage] {beam_id}: {len(beam_done)} stage1 success → {len(top_n)} RASPA candidates")

    # ===========================================================
    # LOCAL GPU PACMAN-CHARGE STAGE (Ewald adsorbates only, e.g. CO2)
    # ===========================================================
    if _use_pacman:
        print(f"\n[CHARGE] charge_method=Ewald — running PACMAN-charge locally before R2")
        z_stage2_jobs, af_total_stage2_jobs = run_local_charge_stage(
            z_stage2_jobs=z_stage2_jobs,
            af_total_stage2_jobs=af_total_stage2_jobs,
            iter_dir=iter_dir,
            hpc_iter_dir=hpc_iter_dir,
        )
        if not z_stage2_jobs and not any(af_total_stage2_jobs.values()):
            print("[CHARGE] All PACMAN charges failed — aborting R2")
            from core.live_runner import LiveResults
            return LiveResults()

    # ===========================================================
    # ROUND 2: Z-top15 + A/F/total-top10 stage2_only (RASPA only)
    # ===========================================================
    n_aft_r2 = sum(len(v) for v in af_total_stage2_jobs.values())
    print(f"\n[HPC Two-Stage] Round 2: Z-{len(z_stage2_jobs)} + A/F/total-{n_aft_r2} RASPA only")

    r2_manifest_path = prepare_r2_manifest(
        z_stage2_jobs=z_stage2_jobs,
        af_total_stage2_jobs=af_total_stage2_jobs,
        iteration=iteration,
        experiment_id=experiment_id,
        iter_dir=iter_dir,
        geometry_filter=geometry_filter,
    )
    with open(r2_manifest_path, "r") as f:
        r2_manifest_data = json.load(f)
    n_r2_jobs = r2_manifest_data["n_jobs"]

    if n_r2_jobs == 0:
        print("[HPC Two-Stage] No R2 jobs")
        from core.live_runner import LiveResults
        return LiveResults()

    _ssh_run(f"mkdir -p {hpc_iter_dir}/results")
    _scp_upload(r2_manifest_path, f"{hpc_iter_dir}/batch_manifest.json")

    print(f"[HPC] Submitting {n_r2_jobs} Round-2 jobs...")
    r2_submit = _ssh_run(
        f"cd {hpc_iter_dir} && JOB_PREFIX={job_prefix} bash {hpc_remote_scripts}/{config.HPC_SUBMIT_SCRIPT} "
        f"batch_manifest.json results {config.HPC_NODE_PROPERTY}{zeo_flag}{ads_flag}",
        timeout=600,
    )
    print(f"[HPC] R2 submit: {(r2_submit.stdout or '').strip()[-200:]}")
    if r2_submit.returncode != 0:
        raise RuntimeError(f"R2 job submission failed (rc={r2_submit.returncode})")

    r2_beam_filenames: dict = defaultdict(list)
    for job in r2_manifest_data["jobs"]:
        r2_beam_filenames[job.get("beam_id", "total")].append(job["filename"])
    _hpc_poll(hpc_iter_dir, f"{hpc_iter_dir}/results", n_r2_jobs, r2_beam_filenames, job_prefix)

    r2_agg = _ssh_run(
        f"cd {hpc_iter_dir} && python {hpc_remote_scripts}/aggregate_results.py "
        f"--manifest batch_manifest.json --results-dir results"
    )
    print(f"[HPC] R2 aggregate: {(r2_agg.stdout or '').strip()[-200:]}")

    local_results_dir = os.path.join(iter_dir, "hpc_results")
    os.makedirs(local_results_dir, exist_ok=True)
    local_results_path = os.path.join(local_results_dir, "batch_results.json")
    _scp_download(f"{hpc_iter_dir}/results/batch_results.json", local_results_path)

    print("[HPC] Parsing Round-2 results...")
    live_results = collect_results(
        results_path=local_results_path,
        sim_cache=sim_cache,
        n_per_beam=config.LIVE_SIM_N_PER_BEAM,
    )
    return live_results


def _hpc_simulate_single(
    beam_pools: dict,
    iteration: int,
    experiment_id: str,
    experiment_dir: str,
    sim_cache,
    use_zeo: bool = False,
    geometry_filter: dict = None,
    hpc_iter_dir: str = None,
    hpc_remote_scripts: str = None,
    zeo_flag: str = "",
    ads_flag: str = "",
    job_prefix: str = "llm4mof",
) -> "LiveResults":
    """Single-stage HPC pipeline (standard flow). Used internally by _hpc_simulate()."""
    from collections import defaultdict

    iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)

    # Step 1: Generate manifest
    manifest_path = prepare_manifest(
        beam_pools=beam_pools,
        iteration=iteration,
        experiment_id=experiment_id,
        iter_dir=iter_dir,
        geometry_filter=geometry_filter,
    )
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
    n_jobs = manifest_data["n_jobs"]

    if n_jobs == 0:
        print("[HPC] No jobs to simulate (all beams empty)")
        from core.live_runner import LiveResults
        return LiveResults()

    # Step 2: Upload
    _ssh_run(f"mkdir -p {hpc_iter_dir}/results")
    _scp_upload(manifest_path, f"{hpc_iter_dir}/batch_manifest.json")

    # Step 3: Submit
    print(f"[HPC] Submitting {n_jobs} jobs via qsub...")
    submit_result = _ssh_run(
        f"cd {hpc_iter_dir} && JOB_PREFIX={job_prefix} bash {hpc_remote_scripts}/{config.HPC_SUBMIT_SCRIPT} "
        f"batch_manifest.json results {config.HPC_NODE_PROPERTY}{zeo_flag}{ads_flag}",
        timeout=600,
    )
    print(f"[HPC] Submit output: {(submit_result.stdout or '').strip()[-200:]}")
    if submit_result.returncode != 0:
        print(f"[HPC] Submit stderr: {(submit_result.stderr or '').strip()[-500:]}")
        raise RuntimeError(f"Job submission failed (rc={submit_result.returncode})")

    # Step 4: Poll
    beam_filenames: dict = defaultdict(list)
    for job in manifest_data["jobs"]:
        beam_filenames[job.get("beam_id", "total")].append(job["filename"])
    _hpc_poll(hpc_iter_dir, f"{hpc_iter_dir}/results", n_jobs, beam_filenames, job_prefix)

    # Step 5: Aggregate on HPC
    print("[HPC] Aggregating results on remote...")
    agg_result = _ssh_run(
        f"cd {hpc_iter_dir} && python {hpc_remote_scripts}/aggregate_results.py "
        f"--manifest batch_manifest.json --results-dir results"
    )
    print(f"[HPC] Aggregate: {(agg_result.stdout or '').strip()[-200:]}")

    # Step 6: Download
    local_results_dir = os.path.join(iter_dir, "hpc_results")
    os.makedirs(local_results_dir, exist_ok=True)
    local_results_path = os.path.join(local_results_dir, "batch_results.json")
    _scp_download(f"{hpc_iter_dir}/results/batch_results.json", local_results_path)

    # Step 7: Parse
    print("[HPC] Parsing results...")
    live_results = collect_results(
        results_path=local_results_path,
        sim_cache=sim_cache,
        n_per_beam=config.LIVE_SIM_N_PER_BEAM,
    )
    return live_results


def _hpc_simulate(
    beam_pools: dict,
    iteration: int,
    experiment_id: str,
    experiment_dir: str,
    sim_cache,
    use_zeo: bool = False,
    geometry_filter: dict = None,
    job_prefix: str = "llm4mof",
) -> "LiveResults":
    """
    Run one iteration's simulations on HPC via SSH polling.

    Single-process flow:
      1. Generate manifest from beam_pools
      2. scp upload manifest + HPC scripts to the HPC host
      3. ssh submit via submit_iteration.sh
      4. Poll via SSH every HPC_POLL_INTERVAL until all .DONE files appear
      5. ssh run aggregate_results.py on HPC
      6. scp download batch_results.json
      7. Parse into LiveResults via collect_results()

    When use_zeo=True, dispatches to _hpc_simulate_two_stage():
      R1 all beams stage1_only → R2 RASPA only on successes.
    """
    iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)

    # Shared HPC path setup
    hpc_base = config.HPC_BASE_DIR
    hpc_exp_dir = f"{hpc_base}/{experiment_id}"
    hpc_iter_dir = f"{hpc_exp_dir}/iter_{iteration}"
    _ssh_run(f"mkdir -p {hpc_iter_dir}")

    # Upload HPC scripts (ensure they're current)
    hpc_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpc")
    hpc_remote_scripts = f"{hpc_base}/hpc"
    _ssh_run(f"mkdir -p {hpc_remote_scripts}")
    for script in ["run_mof_sim.py", "submit_iteration.sh", "submit_iteration_packed.sh", "aggregate_results.py"]:
        script_path = os.path.join(hpc_scripts_dir, script)
        if os.path.exists(script_path):
            _scp_upload(script_path, f"{hpc_remote_scripts}/{script}")

    # Upload config.py so run_mof_sim.py can import LIVE_SIM_ADSORBATE_CONFIGS
    config_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    _scp_upload(config_local, f"{hpc_base}/config.py")

    # Upload forcefield files (required by RASPA3 on the HPC node)
    local_ff_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "simulation", "gcmc", "forcefield")
    remote_ff_root = f"{hpc_base}/forcefields"
    _ssh_run(f"mkdir -p {remote_ff_root}")
    if os.path.isdir(local_ff_root):
        for ff_name in os.listdir(local_ff_root):
            local_ff_dir = os.path.join(local_ff_root, ff_name)
            if os.path.isdir(local_ff_dir):
                _scp_upload_dir(local_ff_dir, remote_ff_root)
        print(f"[HPC] Forcefield files uploaded to {remote_ff_root}")

    # Upload Zeo++ binary if --zeo is enabled
    zeo_flag = " --no-zeo"  # explicit placeholder so $4 is always consumed by submit_iteration.sh
    if use_zeo:
        # Check if Zeo++ already exists on HPC (e.g., installed via conda)
        remote_zeo = f"{hpc_remote_scripts}/network"
        zeo_check = _ssh_run(f"test -x {remote_zeo} && echo EXISTS", check=False)
        if "EXISTS" in (zeo_check.stdout or ""):
            zeo_flag = " --use-zeo"
            print(f"[HPC] Zeo++ binary found on HPC: {remote_zeo}")
        else:
            # Try local download + upload (works on Linux, may fail on Windows)
            try:
                from core.simulation.zeo.install import ensure_zeopp
                zeopp_bin_local = ensure_zeopp()
                _scp_upload(zeopp_bin_local, remote_zeo)
                _ssh_run(f"chmod +x {remote_zeo}")
                zeo_flag = " --use-zeo"
                print(f"[HPC] Zeo++ binary uploaded: {zeopp_bin_local}")
            except (FileNotFoundError, OSError) as e:
                print(f"[HPC] WARNING: Cannot install Zeo++ locally ({e}). "
                      f"Install on HPC: conda install -c conda-forge zeopp-lsmo && "
                      f"ln -sf $(which network) {remote_zeo}")
                raise RuntimeError("Zeo++ binary not available on HPC or locally")

    _ads = getattr(config, "LIVE_SIM_ADSORBATE", "h2")
    _xe_molfrac = getattr(config, "LIVE_SIM_XE_MOLFRAC", 0.20)
    _temp = config.LIVE_SIM_RASPA_TEMPERATURE
    _pres_bar = config.LIVE_SIM_RASPA_PRESSURE / 1e5
    ads_flag = f" --adsorbate {_ads} --xe-molfrac {_xe_molfrac} --temperature {_temp} --pressure {_pres_bar}"

    # Dispatch: two-stage (all beams R1=stage1, R2=RASPA) when --zeo is enabled
    use_two_stage = use_zeo

    if use_two_stage:
        print(f"[HPC] Two-stage pipeline activated (R1: pormake+lammps+zeo, R2: RASPA only)")
        return _hpc_simulate_two_stage(
            beam_pools=beam_pools,
            iteration=iteration,
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            sim_cache=sim_cache,
            geometry_filter=geometry_filter,
            hpc_iter_dir=hpc_iter_dir,
            hpc_remote_scripts=hpc_remote_scripts,
            zeo_flag=zeo_flag,
            ads_flag=ads_flag,
            job_prefix=job_prefix,
        )
    else:
        return _hpc_simulate_single(
            beam_pools=beam_pools,
            iteration=iteration,
            experiment_id=experiment_id,
            experiment_dir=experiment_dir,
            sim_cache=sim_cache,
            use_zeo=use_zeo,
            geometry_filter=geometry_filter,
            hpc_iter_dir=hpc_iter_dir,
            hpc_remote_scripts=hpc_remote_scripts,
            zeo_flag=zeo_flag,
            ads_flag=ads_flag,
            job_prefix=job_prefix,
        )


def run_live_experiment() -> None:
    args = parse_args()

    # Mark live sim mode so feedback_generator uses nlargest (not random sample)
    config._LIVE_SIM_ACTIVE = True

    # --- packed submit override (multiple jobs per submission) ---
    if args.packed:
        config.HPC_SUBMIT_SCRIPT = "submit_iteration_packed.sh"
        print("[System] HPC submit mode: packed (multiple jobs per submission)")

    # --- Smoke test overrides ---
    if args.smoke:
        config.LIVE_SIM_N_PER_BEAM = 1
        config.LIVE_SIM_RASPA_CYCLES = 200
        config.LIVE_SIM_RASPA_INIT_CYCLES = 100
        config.LIVE_SIM_MAX_ITERATIONS = 1
        config.LIVE_SIM_POOL_MULTIPLIER = 2
        config.LIVE_SIM_POOL_MULTIPLIER_RANDOM = 2
        config.LIVE_SIM_Z_POOL_SIZE = 5    # Reduced for smoke tests
        config.LIVE_SIM_Z_RASPA_TOP = 3    # Reduced for smoke tests
        if not args.hpc:
            config.LIVE_SIM_SKIP_LAMMPS = True  # LAMMPS not on Windows (HPC has it)

    if args.iterations:
        config.LIVE_SIM_MAX_ITERATIONS = args.iterations

    # --- Adsorbate config (must come before --pressure override) ---
    ads_cfg = config.LIVE_SIM_ADSORBATE_CONFIGS.get(args.adsorbate, config.LIVE_SIM_ADSORBATE_CONFIGS["h2"])
    config.LIVE_SIM_ADSORBATE = args.adsorbate
    config.LIVE_SIM_XE_MOLFRAC = args.xe_molfrac
    config.LIVE_SIM_RASPA_TEMPERATURE = ads_cfg["temperature"]
    config.LIVE_SIM_RASPA_PRESSURE = ads_cfg["pressure"]
    if args.adsorbate != "h2":
        print(f"[System] Adsorbate: {args.adsorbate.upper()} "
              f"(T={ads_cfg['temperature']}K, P={ads_cfg['pressure']/1e5:.2f} bar, "
              f"ChargeMethod={ads_cfg['charge_method']})")
        if args.adsorbate == "xekr":
            print(f"[System] Xe/Kr mole fractions: Xe={args.xe_molfrac:.2f}, Kr={1-args.xe_molfrac:.2f}")

    if args.server:
        config.HPC_HOST = args.server
        print(f"[System] HPC server override: {args.server}")
    if args.node_prop:
        config.HPC_NODE_PROPERTY = args.node_prop
        print(f"[System] PBS node property override: {args.node_prop}")
    if args.pressure:
        config.LIVE_SIM_RASPA_PRESSURE = args.pressure * 1e5  # bar → Pa
        print(f"[System] Pressure override: {args.pressure} bar ({config.LIVE_SIM_RASPA_PRESSURE:.0f} Pa)")
    if args.temperature:
        config.LIVE_SIM_RASPA_TEMPERATURE = args.temperature
        print(f"[System] Temperature override: {args.temperature} K")

    # mof2zeo ranking margin (A/B auditable): logs which GEOM_MARGIN_MODE this run used.
    print(f"[System] GEOM_MARGIN_MODE: {config.GEOM_MARGIN_MODE} "
          f"(mof2zeo ranking window expansion; train_std=master ±3.2A di, mae=±0.75A, off=strict)")

    # --- Unit variant ---
    if args.pormake_unit == "gperL":
        config._PORMAKE_GPERL_ACTIVE = True
        print("[System] Unit: g/L (volumetric mass concentration)")
    elif args.pormake_unit == "molkg":
        config._PORMAKE_GRAVIMETRIC_ACTIVE = True
        print("[System] Unit: mol/kg (gravimetric)")
    else:
        print("[System] Unit: cm³(STP)/cm³ (volumetric, default)")

    print_banner(smoke=args.smoke)

    # --- Validate API keys ---
    validate_api_keys()

    # --- Get user inquiry ---
    if args.inquiry:
        user_inquiry = args.inquiry
    elif args.resume:
        # Load from previous experiment
        resume_dir = os.path.join(EXPERIMENTS_DIR, args.resume)
        inquiry_file = os.path.join(resume_dir, "raw_user_input.txt")
        if os.path.exists(inquiry_file):
            with open(inquiry_file, "r", encoding="utf-8") as f:
                user_inquiry = f.read().strip()
        else:
            user_inquiry = input("Enter design inquiry: ").strip()
    else:
        print("Enter your MOF design inquiry (or press Enter for default):")
        user_inquiry = input("> ").strip()
        if not user_inquiry:
            user_inquiry = DEFAULT_INQUIRY

    print(f"\n[System] Design Inquiry: {user_inquiry[:200]}...")

    # --- Metric detection (ported from run_experiment.py:290-329) ---
    # Live mode supports H2 via RASPA3 GCMC. RASPA3 natively outputs mol/kg;
    # feedback_live_adapter converts to the active unit based on config flags.
    config.ACTIVE_METRIC_COLUMN = "target"
    _ads_display = {
        "h2": "H2 Uptake", "ch4": "CH4 Uptake", "co2": "CO2 Uptake", "xekr": "Xe/Kr Selectivity",
    }
    inquiry_lower = user_inquiry.lower()

    # Build metric label from actual adsorbate T/P (already applied to config above)
    _temp_k = int(config.LIVE_SIM_RASPA_TEMPERATURE)
    _pres_bar_display = config.LIVE_SIM_RASPA_PRESSURE / 1e5
    _pres_label = f"{int(_pres_bar_display)}bar" if _pres_bar_display == int(_pres_bar_display) else f"{_pres_bar_display:.1f}bar"
    active_metric_name = f"{_ads_display.get(args.adsorbate, 'Uptake')} ({_pres_label} {_temp_k}K)"

    # H2 only: activate 5bar PorMake CSV (word-boundary match avoids "35 bar", "2.5 bar" false positives)
    if args.adsorbate == "h2":
        _is_5bar = (args.pressure and args.pressure <= 10) or \
                   bool(re.search(r'(?<!\d)5\s*bar', inquiry_lower))
        if _is_5bar:
            config._PORMAKE_5BAR_ACTIVE = True
            src = "--pressure" if args.pressure and args.pressure <= 10 else "inquiry"
            print(f"[System] PorMake 5bar pressure mode active (from {src})")

    # Unit variant: CLI override > inquiry keyword detection
    if args.pormake_unit:
        if args.pormake_unit == "gperL":
            config._PORMAKE_GPERL_ACTIVE = True
            print(f"[System] PorMake unit variant override: g/L")
        elif args.pormake_unit == "molkg":
            config._PORMAKE_GRAVIMETRIC_ACTIVE = True
            print(f"[System] PorMake unit variant override: mol/kg")
        else:
            print(f"[System] PorMake unit variant override: volumetric (default)")
    else:
        _UNIT_KEYWORDS = [
            (["g/l", "g per l", "gperl", "g/L", "gram per liter", "grams per liter"], "gperL"),
            (["mol/kg", "gravimetric", "moles per kg"],                                 "molkg"),
        ]
        for unit_kws, unit_type in _UNIT_KEYWORDS:
            if any(kw in inquiry_lower for kw in unit_kws):
                if unit_type == "gperL":
                    config._PORMAKE_GPERL_ACTIVE = True
                    print(f"[System] PorMake unit variant: g/L")
                elif unit_type == "molkg":
                    config._PORMAKE_GRAVIMETRIC_ACTIVE = True
                    print(f"[System] PorMake unit variant: mol/kg")
                break

    # Xe/Kr selectivity is a dimensionless ratio — emit "(dimensionless)" to
    # match markscheme behaviour. Without this guard, get_active_unit() returns
    # the PorMake unit (g/L or cm³(STP)/cm³) and is blindly appended, producing
    # the nonsensical label "Xe/Kr Selectivity (...) (cm³(STP)/cm³)".
    if args.adsorbate == "xekr":
        if "(dimensionless)" not in active_metric_name:
            active_metric_name = f"{active_metric_name} (dimensionless)"
    else:
        unit_label = config.get_active_unit()
        if unit_label and f"({unit_label})" not in active_metric_name:
            active_metric_name = f"{active_metric_name} ({unit_label})"
    print(f"[System] Live mode: metric = {active_metric_name}")
    print(f"[System] Markscheme CSV: {config.get_master_db_path()}")

    # --- T/P mismatch soft-check ---
    _actual_temp_K = config.LIVE_SIM_RASPA_TEMPERATURE
    _actual_pres_bar = config.LIVE_SIM_RASPA_PRESSURE / 1e5
    _tp_warnings = []
    _pres_matches = re.findall(r"(\d+(?:\.\d+)?)\s*bar", user_inquiry, re.IGNORECASE)
    _temp_matches = re.findall(r"(\d+(?:\.\d+)?)\s*K\b", user_inquiry)
    for _p in _pres_matches:
        _p_val = float(_p)
        if abs(_p_val - _actual_pres_bar) / max(_actual_pres_bar, 1) > 0.05:
            _tp_warnings.append(
                f"  Pressure: inquiry mentions {_p_val} bar, but simulation will run at {_actual_pres_bar} bar"
            )
    for _t in _temp_matches:
        _t_val = float(_t)
        if abs(_t_val - _actual_temp_K) > 2.0:
            _tp_warnings.append(
                f"  Temperature: inquiry mentions {_t_val} K, but simulation will run at {_actual_temp_K} K"
            )
    if _tp_warnings:
        print("\n[WARNING] T/P mismatch detected between inquiry and simulation config:")
        for _w in _tp_warnings:
            print(_w)
        print("  → Use --pressure and/or --temperature to align. Continuing in 5s... (Ctrl+C to abort)")
        time.sleep(5)

    # --- Create/resume experiment directory ---
    if args.resume:
        experiment_dir = os.path.join(EXPERIMENTS_DIR, args.resume)
        if not os.path.isdir(experiment_dir):
            print(f"[ERROR] Resume directory not found: {experiment_dir}")
            sys.exit(1)
        print(f"[System] Resuming experiment: {experiment_dir}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        mode_label = "smoke" if args.smoke else "live"
        experiment_dir = os.path.join(
            EXPERIMENTS_DIR, f"exp_{timestamp}_{mode_label}"
        )
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"[System] Experiment directory: {experiment_dir}")

    # Save raw input for reproducibility
    with open(
        os.path.join(experiment_dir, "raw_user_input.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(user_inquiry)

    # --- Initialize components ---
    print("\n[System] Initializing components...")
    usage_log_path = os.path.join(experiment_dir, "usage_log.json")
    agent1 = Agent1Handler(usage_log_path=usage_log_path)
    agent2 = Agent2Handler(usage_log_path=usage_log_path)
    matchmaker = Matchmaker()
    feedback_gen = FeedbackGenerator()
    feedback_gen.experiment_dir = experiment_dir
    memory = MemoryManager(
        experiment_dir, user_inquiry, model_name=ACTIVE_MODEL
    )
    logger = ExperimentLogger(experiment_dir)
    logger.log_model_info(ACTIVE_MODEL)
    logger.log_user_inquiry(user_inquiry)

    # --- Simulation cache ---
    cache_path = os.path.join(experiment_dir, "sim_cache.jsonl")
    sim_cache = SimCache(cache_path)

    # --- Determine starting iteration (for resume) ---
    start_iter = 1
    if args.resume:
        existing = [
            d for d in os.listdir(experiment_dir)
            if d.startswith("iter_") and os.path.isdir(
                os.path.join(experiment_dir, d)
            )
        ]
        if existing:
            last_iter = max(int(d.split("_")[1]) for d in existing)
            if args.collect:
                # --collect: resume at the iteration with a checkpoint
                start_iter = last_iter
            else:
                start_iter = last_iter + 1
            print(f"[System] Resuming from iteration {start_iter} "
                  f"(cache: {len(sim_cache)} entries)")

    # --- ITERATION LOOP ---
    current_hypothesis = None
    current_feedback = ""
    max_iter = config.LIVE_SIM_MAX_ITERATIONS

    # real_iterations: counts only iterations where RASPA actually ran.
    # matchmaker failure and geometry abort do NOT count against the budget.
    # iteration: absolute counter used for directory naming (always increments).
    iteration = start_iter - 1
    # On resume, count already-completed real iterations (those with R2 RASPA results).
    if args.resume:
        real_iterations = sum(
            1 for d in os.listdir(experiment_dir)
            if d.startswith("iter_") and os.path.exists(
                os.path.join(experiment_dir, d, "hpc_results", "batch_results.json")
            )
        )
        print(f"[System] Resuming: {real_iterations} real iterations already completed, "
              f"{max(0, max_iter - real_iterations)} remaining.")
    else:
        real_iterations = 0
    _MAX_TOTAL_ATTEMPTS = max_iter * 3  # safety cap to prevent infinite loop

    while real_iterations < max_iter and (iteration - start_iter + 1) < _MAX_TOTAL_ATTEMPTS:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"         LIVE ITERATION {iteration}  (real {real_iterations + 1}/{max_iter})")
        print(f"{'='*60}")

        logger.log_iteration_start(iteration)

        # ----- STEP A/B: LLM AGENTS (skip if --collect with checkpoint) -----
        if args.collect:
            # Load checkpoint from --prepare phase
            iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
            ckpt_path = os.path.join(iter_dir, "checkpoint.json")
            if os.path.exists(ckpt_path):
                print(f"[Collect] Loading checkpoint: {ckpt_path}")
                try:
                    with open(ckpt_path, "r", encoding="utf-8") as f:
                        ckpt = json.load(f)
                    current_hypothesis = ckpt["hypothesis"]
                    constraints = ckpt["constraints"]
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[ERROR] Checkpoint corrupted: {ckpt_path}")
                    print(f"[ERROR] {type(e).__name__}: {e}")
                    print(f"[HINT] Delete the checkpoint and re-run --prepare, "
                          f"or restore from a backup.")
                    break
            else:
                print(f"[ERROR] No checkpoint at {ckpt_path}. Run --prepare first.")
                break
        else:
            # ----- STEP A: AGENT 1 — GENERATE/REFINE HYPOTHESIS -----
            # Restore handoff state from previous --collect (feedback + conversation)
            if iteration > 1 and not current_feedback:
                handoff_path = os.path.join(experiment_dir, "handoff_state.json")
                if os.path.exists(handoff_path):
                    print(f"[Handoff] Restoring Agent 1 state from {handoff_path}")
                    try:
                        with open(handoff_path, "r", encoding="utf-8") as f:
                            handoff = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Handoff state corrupted: {e}")
                        print(f"[HINT] Delete {handoff_path} to start fresh (loses conversation history).")
                        handoff = {}
                    current_feedback = handoff.get("feedback", "")
                    conv_history = handoff.get("conversation_history", [])
                    if conv_history:
                        agent1.set_conversation_history(conv_history)
                        print(f"[Handoff] Restored {len(conv_history)} conversation turns")

            if iteration == 1 and not current_feedback:
                current_hypothesis = agent1.generate_initial_hypothesis(user_inquiry)
            else:
                current_hypothesis = agent1.refine_hypothesis(current_feedback)

            if not current_hypothesis:
                print("[ERROR] Agent 1 failed to generate hypothesis. Exiting.")
                break

            logger.log_hypothesis(current_hypothesis)

            # ----- STEP B: AGENT 2 — EXTRACT CONSTRAINTS -----
            constraints = agent2.extract_constraints(current_hypothesis)
            if not constraints:
                print("[ERROR] Agent 2 failed to extract constraints. Exiting.")
                break

            logger.log_constraints(constraints)

        # ----- STEP B.5: VALIDATION (same as run_experiment.py) -----
        global_reqs = constraints.get("global_requirements", {})
        exclude_tags = set(global_reqs.get("exclude_tags", []))
        include_tags = set(global_reqs.get("include_tags", []))
        linker_fgs = set(
            constraints.get("linker_query", {}).get("functional_groups", [])
        )

        for tag in include_tags & exclude_tags:
            constraints["global_requirements"]["exclude_tags"].remove(tag)
        for tag in linker_fgs & exclude_tags:
            constraints["global_requirements"]["exclude_tags"].remove(tag)

        # ----- STEP C: SIMULATION (local, HPC auto, or HPC manual) -----

        # Build beam pools (needed for --hpc and --prepare)
        if args.hpc or args.prepare:
            print("\n[HPC] Building candidate pools...")
            # Two-stage mode (--zeo): Z=100, A/F/total=30 for stage1
            beam_pools = prepare_beam_pools(
                specs=constraints,
                matchmaker=matchmaker,
                n_per_beam=config.LIVE_SIM_N_PER_BEAM,
                z_pool_size=config.LIVE_SIM_Z_POOL_SIZE if args.zeo else None,
                af_pool_size=config.LIVE_SIM_AF_POOL_SIZE if args.zeo else None,
            )

            # Collect matchmaker stats for logging (Fix 1.2 + 1.3)
            matchmaker_summary = {}
            for bid, pool in beam_pools.items():
                topos = set()
                nodes = set()
                edges = set()
                for rm in pool:
                    topos.add(rm.component.topology)
                    nodes.add(rm.component.node)
                    edges.add(rm.component.edge)
                matchmaker_summary[bid] = {
                    "n_candidates": len(pool),
                    "n_topologies": len(topos),
                    "n_nodes": len(nodes),
                    "n_edges": len(edges),
                }
            total_candidates = sum(v["n_candidates"] for v in matchmaker_summary.values())
            print(f"[HPC] Total candidates across beams: {total_candidates}")

            # Log matchmaker results (Fix 1.3)
            mm_log = {
                "topology": list({rm.component.topology for p in beam_pools.values() for rm in p}),
                "node": list({rm.component.node for p in beam_pools.values() for rm in p}),
                "edge": list({rm.component.edge for p in beam_pools.values() for rm in p}),
            }
            logger.log_matchmaker_results(mm_log)

            # --- MATCHMAKER FAILURE SHORTCUT ---
            # If all hypothesis beams (Z, A, F) found 0 candidates, skip HPC entirely
            # and feed the failure back to Agent 1 immediately.
            hypothesis_beams_empty = all(
                matchmaker_summary.get(bid, {}).get("n_candidates", 0) == 0
                for bid in ["Z", "A", "F"]
            )
            if hypothesis_beams_empty:
                print(f"\n[HPC] WARNING: All hypothesis beams (Z/A/F) returned 0 matchmaker candidates.")
                print(f"[HPC] Skipping HPC simulation — sending constraint failure feedback to Agent 1.")

                node_q = constraints.get("node_query", {})
                metals = node_q.get("metals_include", [])
                cn_requested = node_q.get("connectivity", [])
                node_abs = node_q.get("abstract_features", {})
                current_feedback = _build_matchmaker_failure_feedback(
                    metals, cn_requested, node_abs,
                )

                iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
                os.makedirs(iter_dir, exist_ok=True)
                with open(os.path.join(iter_dir, "agent1_output.json"), "w", encoding="utf-8") as f:
                    json.dump(current_hypothesis, f, indent=2, ensure_ascii=False)
                with open(os.path.join(iter_dir, "agent2_output.json"), "w", encoding="utf-8") as f:
                    json.dump(constraints, f, indent=2, ensure_ascii=False)
                feedback_path = os.path.join(iter_dir, "feedback_selected.txt")
                with open(feedback_path, "w", encoding="utf-8") as f:
                    f.write("Feedback Type: Matchmaker Failure (No HPC Run)\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(current_feedback)

                print(f"[HPC] Failure feedback saved → {feedback_path}")
                print(f"[HPC] Proceeding to next iteration with revised constraints...")
                continue  # Skip HPC entirely — loop back to Agent 1

        if args.hpc:
            # HPC AUTO MODE: single-process SSH polling
            experiment_id = os.path.basename(experiment_dir)

            # Save checkpoint so --collect can resume if process is interrupted
            iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            ckpt_path = os.path.join(iter_dir, "checkpoint.json")
            if not os.path.exists(ckpt_path):
                checkpoint = {
                    "iteration": iteration,
                    "hypothesis": current_hypothesis,
                    "constraints": constraints,
                    "experiment_dir": experiment_dir,
                }
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            live_results = _hpc_simulate(
                beam_pools=beam_pools,
                iteration=iteration,
                experiment_id=experiment_id,
                experiment_dir=experiment_dir,
                sim_cache=sim_cache,
                use_zeo=args.zeo,
                geometry_filter=constraints.get("geometry_filter", {}) or {},
                job_prefix=args.job_prefix,
            )

            if args.zeo:
                _apply_zeo_postprocess(live_results)

            # --- GEOMETRY ABORT: 0 MOFs passed agent1 geometry filter in stage1 ---
            if live_results.geometry_aborted:
                print(f"\n[HPC] GEOMETRY ABORT: no MOFs passed geometry filter in stage1.")
                print(f"[HPC] Sending geometry redesign request to Agent 1.")

                iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
                os.makedirs(iter_dir, exist_ok=True)
                with open(os.path.join(iter_dir, "agent1_output.json"), "w", encoding="utf-8") as f:
                    json.dump(current_hypothesis, f, indent=2, ensure_ascii=False)
                with open(os.path.join(iter_dir, "agent2_output.json"), "w", encoding="utf-8") as f:
                    json.dump(constraints, f, indent=2, ensure_ascii=False)

                current_feedback = _build_geometry_abort_feedback(
                    live_results.stage1_geometry_stats, constraints
                )
                feedback_path = os.path.join(iter_dir, "feedback_selected.txt")
                with open(feedback_path, "w", encoding="utf-8") as f:
                    f.write("Feedback Type: Geometry Abort (No RASPA Run)\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(current_feedback)

                logger.log_feedback_selection("Geometry Abort", current_feedback)
                print(f"[HPC] Geometry abort feedback saved → {feedback_path}")

                continue  # Back to Agent 1 with geometry redesign request

        elif args.prepare:
            # HPC MANUAL MODE: prepare manifest and exit
            iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
            experiment_id = os.path.basename(experiment_dir)
            manifest_path = prepare_manifest(
                beam_pools=beam_pools,
                iteration=iteration,
                experiment_id=experiment_id,
                iter_dir=iter_dir,
                geometry_filter=constraints.get("geometry_filter", {}) or {},
            )

            # Save checkpoint for --collect phase
            checkpoint = {
                "iteration": iteration,
                "hypothesis": current_hypothesis,
                "constraints": constraints,
                "manifest_path": manifest_path,
                "experiment_dir": experiment_dir,
            }
            ckpt_path = os.path.join(iter_dir, "checkpoint.json")
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            print(f"\n[HPC Prepare] Manifest: {manifest_path}")
            print(f"[HPC Prepare] Checkpoint: {ckpt_path}")
            print(f"\n[ACTION] Upload manifest to HPC and submit jobs.")
            print(f"  Or use: bash scripts/autonomous_loop.sh {experiment_id}")
            break  # Exit — resume with --collect after HPC completes

        elif args.collect:
            # HPC MODE: parse downloaded results
            iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
            results_path = os.path.join(iter_dir, "hpc_results", "batch_results.json")

            if not os.path.exists(results_path):
                print(f"[ERROR] Results not found: {results_path}")
                print("Download batch_results.json from HPC first.")
                break

            print(f"\n[HPC Collect] Parsing results from {results_path}")
            live_results = collect_results(
                results_path=results_path,
                sim_cache=sim_cache,
                n_per_beam=config.LIVE_SIM_N_PER_BEAM,
            )

            if args.zeo:
                _apply_zeo_postprocess(live_results)

        else:
            # LOCAL MODE: run simulation directly
            print("\n[Live Runner] Starting live simulation iteration...")

            live_results = run_live_iteration(
                specs=constraints,
                matchmaker=matchmaker,
                iteration=iteration,
                run_dir=experiment_dir,
                sim_cache=sim_cache,
                n_per_beam=config.LIVE_SIM_N_PER_BEAM,
                use_zeo=args.zeo,
            )

            if args.zeo:
                _apply_zeo_postprocess(live_results)

            # --- GEOMETRY ABORT: 0 MOFs passed agent1 geometry filter in stage1 ---
            if live_results.geometry_aborted:
                print(f"\n[Live] GEOMETRY ABORT: no MOFs passed geometry filter in stage1.")
                print(f"[Live] Sending geometry redesign request to Agent 1.")

                iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
                os.makedirs(iter_dir, exist_ok=True)
                with open(os.path.join(iter_dir, "agent1_output.json"), "w", encoding="utf-8") as f:
                    json.dump(current_hypothesis, f, indent=2, ensure_ascii=False)
                with open(os.path.join(iter_dir, "agent2_output.json"), "w", encoding="utf-8") as f:
                    json.dump(constraints, f, indent=2, ensure_ascii=False)

                current_feedback = _build_geometry_abort_feedback(
                    live_results.stage1_geometry_stats, constraints
                )
                feedback_path = os.path.join(iter_dir, "feedback_selected.txt")
                with open(feedback_path, "w", encoding="utf-8") as f:
                    f.write("Feedback Type: Geometry Abort (No RASPA Run)\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(current_feedback)

                logger.log_feedback_selection("Geometry Abort", current_feedback)
                print(f"[Live] Geometry abort feedback saved → {feedback_path}")

                agent1.add_feedback(current_feedback)
                continue  # Back to Agent 1 with geometry redesign request

        # Check for catastrophic failure — only count beams that had
        # simulation failures (pool exhausted), NOT matchmaker empty-results
        # (which are normal for over-constrained first hypotheses).
        sim_aborted = [
            b for b in live_results.aborted_beams
            if b in live_results.beams and live_results.beams[b].pool_size > 0
        ]
        if len(sim_aborted) >= 3:
            print(f"\n[WARNING] {len(sim_aborted)} beams failed during simulation: "
                  f"{sim_aborted}")
            print("Too many simulation failures. Stopping experiment.")
            break
        if live_results.aborted_beams:
            mm_empty = [b for b in live_results.aborted_beams if b not in sim_aborted]
            if mm_empty:
                print(f"\n[INFO] Beams {mm_empty} had no matchmaker candidates "
                      f"(over-constrained hypothesis — normal for iteration 1)")

        # ----- SAVE PER-ITERATION FILES -----
        iter_dir = os.path.join(experiment_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        agent1_path = os.path.join(iter_dir, "agent1_output.json")
        with open(agent1_path, "w", encoding="utf-8") as f:
            json.dump(current_hypothesis, f, indent=2, ensure_ascii=False)

        agent2_path = os.path.join(iter_dir, "agent2_output.json")
        with open(agent2_path, "w", encoding="utf-8") as f:
            json.dump(constraints, f, indent=2, ensure_ascii=False)

        # ----- STEP D: FEEDBACK GENERATION -----
        # Convert live results → filter_sets format → feedback type 1 (4-beam)
        filter_sets = live_results_to_filter_sets(live_results)

        # Detect if geometry_filter was null (all fields None) → prevents spurious geometry comparison
        gf = constraints.get("geometry_filter", {}) or {}
        geometry_null = not any(v is not None for v in gf.values())

        feedback_type = 1  # Always 4-beam diagnostic in live mode
        current_feedback = feedback_gen.generate_feedback(
            feedback_type, filter_sets, metric_name=active_metric_name,
            geometry_null=geometry_null
        )

        print(f"\n--- Feedback Preview ---")
        preview = current_feedback[:500]
        print(preview + "..." if len(current_feedback) > 500 else preview)

        # Save feedback
        feedback_path = os.path.join(iter_dir, "feedback_selected.txt")
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write("Feedback Type: 4-Beam Diagnostic (Live Simulation)\n")
            f.write("=" * 50 + "\n\n")
            f.write(current_feedback)

        logger.log_feedback_selection("4-Beam Diagnostic (Live)", current_feedback)

        # ----- STEP E: MEMORY -----
        live_summary = {
            "n_simulations": live_results.n_real_simulations,
            "n_failures": live_results.n_failures,
            "wall_clock_s": live_results.wall_clock_seconds,
            "aborted_beams": live_results.aborted_beams,
            "cache_size": len(sim_cache),
        }

        # Build matchmaker_result with real counts (Fix 1.2)
        if args.hpc or args.prepare:
            # matchmaker_summary was computed during beam pool construction
            mm_result = {
                "live_mode": True,
                "topology": list({rm.component.topology for p in beam_pools.values() for rm in p}),
                "node": list({rm.component.node for p in beam_pools.values() for rm in p}),
                "edge": list({rm.component.edge for p in beam_pools.values() for rm in p}),
                "per_beam": matchmaker_summary,
                "summary": live_summary,
            }
        else:
            mm_result = {"live_mode": True, "summary": live_summary}

        memory.add_iteration(
            iteration_num=iteration,
            hypothesis=current_hypothesis,
            constraints=constraints,
            matchmaker_result=mm_result,
            feedback_type="4-Beam Diagnostic (Live)",
            feedback_content=current_feedback,
            sensitivity_summary=live_summary,
        )

        memory.save_conversation_history(agent1.get_conversation_history())

        # ----- SAVE CROSS-ITERATION STATE (for --prepare/--collect handoff) -----
        # This enables the next --prepare to restore Agent 1's context
        handoff_path = os.path.join(experiment_dir, "handoff_state.json")
        with open(handoff_path, "w", encoding="utf-8") as f:
            json.dump({
                "last_iteration": iteration,
                "feedback": current_feedback,
                "conversation_history": agent1.get_conversation_history(),
            }, f, indent=2, ensure_ascii=False)
        print(f"[System] Saved handoff state for next iteration")

        real_iterations += 1
        print(f"\n[System] Iteration {iteration} complete. "
              f"Real {real_iterations}/{max_iter} "
              f"({live_results.wall_clock_seconds:.0f}s, "
              f"cache={len(sim_cache)})")

    # --- EXPERIMENT END ---
    logger.log_experiment_end(iteration)

    print(f"\n{'='*60}")
    print("LIVE EXPERIMENT COMPLETE")
    print(f"Real Iterations: {real_iterations}/{max_iter}")
    print(f"Total Attempts (incl. failures): {iteration - start_iter + 1}")
    print(f"Simulation Cache: {len(sim_cache)} entries")
    print(f"Experiment Dir: {experiment_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_live_experiment()
