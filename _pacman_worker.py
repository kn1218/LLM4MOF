"""One PACMAN DDEC6 charge assignment, in its own process.

Deliberately a separate top-level module rather than a closure in
run_live_experiment.py: ProcessPoolExecutor on Windows uses spawn, so the child
re-imports whatever module holds the callable. Importing run_live_experiment
would drag in torch, config, the matchmaker and the mof2zeo model in every
worker. This module imports PACMANCharge and nothing else.

Each worker is capped to a couple of BLAS threads. torch defaults to using every
core for one inference, so N unconstrained workers would oversubscribe a 20-core
box and run slower than the serial version they replace.
"""
import os

# Must be set before torch is imported anywhere in this process.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")


def charge_one(local_cif: str, timeout_s: int) -> dict:
    """Run PACMAN on *local_cif*. Returns a dict, never raises.

    {"ok": bool, "out": str|None, "elapsed": float, "error": str}
    The caller decides what a failure means; this only reports.
    """
    import threading
    import time

    import torch
    try:
        torch.set_num_threads(2)
    except Exception:
        pass
    from PACMANCharge import pmcharge

    out = local_cif.replace(".cif", "_pacman.cif")
    state = {"ok": False, "error": ""}
    t0 = time.time()

    def _work():
        try:
            pmcharge.predict(cif_file=local_cif, charge_type="DDEC6", digits=6,
                             atom_type=True, neutral=True, keep_connect=True)
            state["ok"] = os.path.isfile(out)
        except Exception as exc:                     # noqa: BLE001 - reported, not raised
            state["error"] = str(exc)[:200]

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    elapsed = time.time() - t0

    if t.is_alive():
        return {"ok": False, "out": None, "elapsed": elapsed, "error": f"timeout {timeout_s}s"}
    if not state["ok"]:
        if os.path.isfile(out):
            try:
                os.remove(out)
            except OSError:
                pass
        return {"ok": False, "out": None, "elapsed": elapsed,
                "error": state["error"] or "no output produced"}
    return {"ok": True, "out": out, "elapsed": elapsed, "error": ""}
