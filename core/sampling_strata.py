"""Metal-stratified sampling (productionized universal lever).

Round-robins the feedback (markscheme) or simulation (live) candidate slots across the METALS present in
the matched beam, so rare-but-best metals (e.g. In 2.6 %, Ga 0.4 % of the PORMAKE space) are not crowded out
by uniform sampling, whose P(seen) is proportional to DB frequency.

FIREWALL: the stratification key is metal IDENTITY only (node BB metadata / index lookup, query-side). It
NEVER reads `target`/performance. The within-metal draw is random. Displayed results remain the sampled rows'
own targets. No full-matched-set statistic is used. → firewall-clean, mode-universal, no-injection (every
present metal gets fair representation; no metal is favored; no RE value hardcoded).

Validated in research/top1&0.1 as the only fully cross-app-validated universal lever. Gated by
config.STRATIFIED_SAMPLING; with the flag OFF this module is never called and behavior is unchanged.

FAIL-SAFE: `stratified_reduce` returns the input df unchanged on any error, so a load/parse failure silently
falls back to the caller's original (uniform) sampling rather than breaking the run.
"""
import json
import random

import config

_bb = None
_hmof_metal = None
_qmof_metal = None


def _load_bb():
    global _bb
    if _bb is None:
        try:
            _bb = {x["ID"]: x for x in json.load(open(config.BB_DICTIONARY_PATH, encoding="utf-8"))}
        except Exception:
            _bb = {}
    return _bb


def _load_index_metals(path, idkey):
    out = {}
    try:
        for r in json.load(open(path, encoding="utf-8")):
            metals = r.get("metals") or []
            out[r.get(idkey)] = "+".join(metals) if metals else "organic"
    except Exception:
        pass
    return out


def metal_of(filename) -> str:
    """DB-agnostic metal identity for a candidate (IDENTITY only; firewall-safe).

    PORMAKE: parse the node BB id (N*) from the filename and look up its metals.
    hMOF/QMOF: look up the id in the index (lazy-loaded). Returns "?" on miss/error.
    """
    global _hmof_metal, _qmof_metal
    try:
        fn = str(filename)
        if fn.startswith("hMOF"):
            if _hmof_metal is None:
                _hmof_metal = _load_index_metals(config.HMOF_INDEX_PATH, "hmof_id")
            return _hmof_metal.get(fn, "?")
        if fn.startswith("qmof"):
            if _qmof_metal is None:
                _qmof_metal = _load_index_metals(config.QMOF_INDEX_PATH, "qmof_id")
            return _qmof_metal.get(fn, "?")
        bb = _load_bb()
        node = next((p for p in fn.split("+")[1:] if p.startswith("N")), None)
        if node is None:
            return "?"
        metals = bb.get(node, {}).get("metals") or []
        return "+".join(metals) if metals else "organic"
    except Exception:
        return "?"


def stratified_reduce(df, n: int):
    """Return a <=n metal-diverse subset of df (round-robin across present metals).

    Firewall: stratifies on metal IDENTITY only; within-metal order shuffled; `target` never consulted.
    Fail-safe: returns df unchanged on any error (caller then falls back to its original sampling).
    """
    try:
        if df is None or len(df) <= n or "filename" not in df.columns:
            return df
        groups = {}
        for idx, fn in df["filename"].items():
            groups.setdefault(metal_of(fn), []).append(idx)
        order = list(groups.keys())
        for g in order:
            random.shuffle(groups[g])
        chosen, gi, guard = [], 0, 0
        while len(chosen) < n and any(groups.values()) and guard < 1_000_000:
            g = order[gi % len(order)]
            if groups[g]:
                chosen.append(groups[g].pop())
            gi += 1
            guard += 1
        return df.loc[chosen]
    except Exception:
        return df


def stratified_select_dicts(items: list, n: int) -> list:
    """Select <=n items from an ordered list of dicts (each with a 'filename' key), metal-diverse.

    Used for stage1→stage2 RASPA candidate selection. Preserves within-metal order (best-first).
    Fail-safe: returns items[:n] on any error.
    """
    try:
        if not items or len(items) <= n:
            return items[:n]
        groups = {}
        for item in items:
            fn = item.get("filename", "")
            groups.setdefault(metal_of(fn), []).append(item)
        order = list(groups.keys())
        chosen, gi, guard = [], 0, 0
        while len(chosen) < n and any(groups.values()) and guard < 1_000_000:
            g = order[gi % len(order)]
            if groups[g]:
                chosen.append(groups[g].pop(0))
            gi += 1
            guard += 1
        return chosen
    except Exception:
        return items[:n]


def stratified_rank_select(ranked, n: int):
    """Select <=n items from a rank-ordered (best-first) list of RankedMOF objects, metal-diverse.

    Used in LIVE mode to choose WHICH assembled candidates to simulate: round-robin across the metals
    present, taking the highest-ranked remaining candidate of each metal in turn — so rare-but-best metals
    (In/Ga) enter the expensive simulation pool instead of being crowded out by a pure geometry-match rank.
    Rank order is preserved WITHIN each metal.

    Firewall: keys on metal IDENTITY only (component node BB), never on performance. Fail-safe: returns
    ranked[:n] on any error (legacy top-N selection).
    """
    try:
        if not ranked:
            return ranked
        if len(ranked) <= n:
            return ranked[:n]
        groups = {}
        for rm in ranked:  # ranked is already best-first; appending preserves that order within a metal
            comp = getattr(rm, "component", None)
            fn = getattr(comp, "filename", None)
            groups.setdefault(metal_of(fn) if fn else "?", []).append(rm)
        order = list(groups.keys())
        chosen, gi, guard = [], 0, 0
        while len(chosen) < n and any(groups.values()) and guard < 1_000_000:
            g = order[gi % len(order)]
            if groups[g]:
                chosen.append(groups[g].pop(0))   # highest-ranked remaining of this metal
            gi += 1
            guard += 1
        return chosen
    except Exception:
        return ranked[:n]
