# =============================================================================
# LLM4MOF Autonomous System - Memory Ledger
# =============================================================================
# Distilled, bounded knowledge state of the design campaign (Kosmos world-model
# pattern). Separate from MemoryManager:
#   MemoryManager  = raw audit log (what happened, full, human/figures)
#   MemoryLedger   = distilled state (what we KNOW, bounded, fed to Agent 1)
#
# Phase 1 (markscheme, additive): provides an explicit global-best anchor +
# frontier + cross-beam medians, injected ADDITIVELY into Agent 1's feedback
# (the multi-turn transcript is left untouched in this phase).
#
# Shared across both runners; the only mode-specific part is the ingest input
# (markscheme = analyzer.filter_sets, live = LiveResults). global_best/frontier
# are computed from the hypothesis-driven beams (Z/A/F) only — never the random
# baseline (total) — so the markscheme CSV oracle is never leaked to Agent 1.
# =============================================================================

import os
import sys
import json
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.name_resolver import get_name_resolver

# Beams that reflect the agent's own hypothesis (used for global_best/frontier).
# 'total' (random baseline) is deliberately excluded to avoid oracle leakage in
# markscheme mode, where 'total' is a random draw from the known benchmark CSV.
_HYPOTHESIS_BEAMS = ["z", "a", "f"]
_ALL_BEAMS = ["z", "a", "f", "total"]
_GEOMETRY_COLS = ["di", "df", "sa", "vf", "density"]


class MemoryLedger:
    """Distilled, bounded knowledge state of an optimization campaign.

    Holds the global best, a top-K frontier, per-beam medians, a geometry
    envelope, and a per-iteration history digest — all derived from the
    agent's own hypothesis beams. Rendered into a compact, blinding-safe text
    block for Agent 1's context.
    """

    def __init__(self, experiment_dir: str, metric_name: str = "Performance",
                 metric_direction: str = "maximize", metric_type: str = "uptake",
                 frontier_k: int = 10, mode: str = "markscheme"):
        """Initialize (or resume) the ledger.

        Args:
            experiment_dir: Directory for memory_ledger.json.
            metric_name: Human-readable metric name (e.g. "CH4 Uptake").
            metric_direction: "maximize" or "minimize".
            metric_type: "uptake" or "ratio" (informational).
            frontier_k: Number of distinct top structures to retain.
            mode: "markscheme" or "live" (governs the porosity gate).
        """
        self.experiment_dir = experiment_dir
        self.metric_name = metric_name
        self.metric_direction = metric_direction
        self.metric_type = metric_type
        self.frontier_k = frontier_k
        self.mode = mode

        self._resolver = get_name_resolver()
        self.path = os.path.join(experiment_dir, "memory_ledger.json")

        # Distilled state
        self.global_best: Optional[Dict[str, Any]] = None
        self.frontier: List[Dict[str, Any]] = []
        self.geometry_envelope: Dict[str, List[float]] = {}
        self.history: List[Dict[str, Any]] = []

        if os.path.exists(self.path):
            self._load()
            print(f"[MemoryLedger] Resumed from {self.path} "
                  f"({len(self.history)} iters, best={self._fmt(self.global_best)})")
        else:
            print(f"[MemoryLedger] Initialized ({mode} mode, metric={metric_name}, {metric_direction})")

    # ------------------------------------------------------------------ utils
    def _is_better(self, value: float, than: Optional[float]) -> bool:
        """Return True if `value` beats `than` under the active direction."""
        if than is None:
            return True
        return value > than if self.metric_direction == "maximize" else value < than

    def _struct_label(self, row: pd.Series) -> str:
        """Blinding-safe chemistry description from a beam row.

        PorMake filenames ("topo+node+edge") → chemistry names via NameResolver.
        hMOF/QMOF → metals + topology + backbone groups (mirrors
        feedback_generator's row descriptions). Never includes database identity
        or raw IDs.
        """
        filename = row.get("filename", "")
        try:
            if isinstance(filename, str) and filename.count("+") == 2:
                return self._resolver.translate_mof_filename(filename)
        except Exception:
            pass
        # Whole-MOF modes (hMOF/QMOF): build from chemistry columns
        parts: List[str] = []
        metals = row.get("metals")
        if isinstance(metals, list) and metals:
            parts.append("/".join(str(m) for m in metals[:3]))
        elif isinstance(metals, str) and metals and metals != "nan":
            parts.append(metals)
        topo = row.get("topology")
        if isinstance(topo, str) and topo and topo != "nan":
            parts.append(topo)
        fg = row.get("functional_groups_categorized")
        if isinstance(fg, dict):
            # Mirror feedback_generator._describe_hmof_row EXACTLY: backbone AND
            # substituents, in Bkbn:[...]/Subs:[...] form. The substituents are
            # what distinguish a winning *combination* (e.g. acetylene+methyl on
            # an aromatic backbone) from the bare family — dropping them makes the
            # anchor lossy and the agent scatters the tags across separate branches.
            backbone = fg.get("backbone", [])
            substituents = fg.get("substituents", [])
            if "carboxylate" in backbone and "carboxyl_any" in backbone:
                backbone = [g for g in backbone if g != "carboxyl_any"]
            if backbone:
                parts.append("Bkbn:[" + ",".join(str(g) for g in backbone[:4]) + "]")
            if substituents:
                parts.append("Subs:[" + ",".join(str(g) for g in substituents[:3]) + "]")
        else:
            # QMOF stores a flat functional_groups list + node geometry
            fg_list = row.get("functional_groups")
            if isinstance(fg_list, list) and fg_list:
                parts.append("[" + ",".join(str(g) for g in fg_list[:3]) + "]")
            geom = row.get("geometry")
            if isinstance(geom, str) and geom and geom not in ("nan", "Unknown"):
                parts.append(f"({geom})")
        return " ".join(parts) if parts else "structure"

    @staticmethod
    def _geometry_of(row: pd.Series) -> Dict[str, float]:
        """Extract available geometry descriptors from a beam row."""
        geo = {}
        for col in _GEOMETRY_COLS:
            if col in row and pd.notna(row[col]):
                try:
                    geo[col] = round(float(row[col]), 4)
                except (ValueError, TypeError):
                    pass
        return geo

    def _passes_gate(self, row: pd.Series) -> bool:
        """Porosity / validity gate.

        Live mode: reject nonporous false positives (VF<=0 or SA<=0) — these
        produce fake-high values, especially for ratio metrics (B-1).
        Markscheme: values are pre-computed from a curated DB, always valid.
        """
        if self.mode != "live":
            return True
        vf = row.get("vf")
        sa = row.get("sa")
        try:
            return float(vf) > 0 and float(sa) > 0
        except (ValueError, TypeError):
            return False

    def _fmt(self, entry: Optional[Dict[str, Any]]) -> str:
        if not entry:
            return "n/a"
        return f"{entry['value']:.2f} (iter {entry['iter']})"

    # ----------------------------------------------------------------- ingest
    def ingest(self, filter_sets: Dict[str, pd.DataFrame], iteration: int,
               hypothesis_id: str, hypothesis: Optional[Dict[str, Any]] = None,
               parent_id: Optional[str] = None) -> None:
        """Update distilled state from this iteration's beams (shared core).

        global_best / frontier / geometry_envelope are derived from the
        hypothesis beams (Z/A/F) only. Per-beam medians (incl. total) are
        recorded for the cross-beam comparison. Each retained structure carries
        provenance {iter, beam, hypothesis_id}.
        """
        # --- collect hypothesis-beam candidates (gated) ---
        candidates: List[Dict[str, Any]] = []
        for beam in _HYPOTHESIS_BEAMS:
            df = filter_sets.get(beam)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty or "target" not in df.columns:
                continue
            for _, row in df.iterrows():
                if pd.isna(row.get("target")) or not self._passes_gate(row):
                    continue
                candidates.append({
                    "value": round(float(row["target"]), 4),
                    "iter": iteration,
                    "beam": beam.upper(),
                    "hypothesis_id": hypothesis_id,
                    "structure": self._struct_label(row),
                    "filename": row.get("filename", ""),
                    "geometry": self._geometry_of(row),
                })

        # --- per-beam medians (incl. total) for cross-beam comparison ---
        beam_medians: Dict[str, Optional[float]] = {}
        for beam in _ALL_BEAMS:
            df = filter_sets.get(beam)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty and "target" in df.columns:
                vals = [float(v) for v in df["target"] if pd.notna(v)]
                beam_medians[beam] = round(float(pd.Series(vals).median()), 4) if vals else None
            else:
                beam_medians[beam] = None

        # --- update global_best (single best) ---
        best_this_iter = None
        for cand in candidates:
            if best_this_iter is None or self._is_better(cand["value"], best_this_iter["value"]):
                best_this_iter = cand
            if self.global_best is None or self._is_better(cand["value"], self.global_best["value"]):
                self.global_best = cand

        # --- update top-K portfolio (frontier) + count how many entered it ---
        # The objective is a strong top-K portfolio, not a single best. We track
        # how many of THIS iteration's candidates newly enter the top-K.
        prev_keys = {(e.get("filename") or e.get("structure")) for e in self.frontier}
        self._update_frontier(candidates)
        self._update_geometry_envelope()
        front_keys = {(e.get("filename") or e.get("structure")) for e in self.frontier}
        iter_keys = {(c.get("filename") or c.get("structure")) for c in candidates}
        new_entries = len((front_keys - prev_keys) & iter_keys)

        # --- verdict + history digest (portfolio-centric) ---
        if best_this_iter is None:
            verdict = "no candidates"
        elif new_entries > 0:
            verdict = f"+{new_entries} to top-{self.frontier_k}"
        else:
            verdict = "no top-K improvement"
        cutoff = self._topk_cutoff()
        tkmed = self._topk_median()
        # Evidence-graft signal (§8-7): a different family beating my CURRENT
        # per-iter quality (this-iter Z median) in the random beam. Per-iter
        # reference (not cumulative cutoff) so a stuck search keeps signalling.
        outside_promising = self._detect_outside_promising(
            filter_sets, beam_medians.get("z"), self._topk_token_set())
        self.history.append({
            "iter": iteration,
            "hypothesis_id": hypothesis_id,
            "parent": parent_id,
            "best_this_iter": round(best_this_iter["value"], 4) if best_this_iter else None,
            "topk_cutoff": round(cutoff, 4) if cutoff is not None else None,
            "topk_median": round(tkmed, 4) if tkmed is not None else None,
            "new_entries": new_entries,
            "outside_promising": outside_promising,
            "beam_medians": beam_medians,
            "verdict": verdict,
        })

    def ingest_markscheme(self, filter_sets: Dict[str, pd.DataFrame], iteration: int,
                          hypothesis_id: str, hypothesis: Optional[Dict[str, Any]] = None,
                          parent_id: Optional[str] = None) -> None:
        """Markscheme adapter. Oracle-safe: only matched Z/A/F beams feed
        global_best (never the raw CSV, never the random 'total' beam)."""
        self.ingest(filter_sets, iteration, hypothesis_id, hypothesis, parent_id)

    def ingest_live(self, live_results: Any, iteration: int, hypothesis_id: str,
                    hypothesis: Optional[Dict[str, Any]] = None,
                    parent_id: Optional[str] = None) -> None:
        """Live adapter (Phase 4). Converts LiveResults beams to the same
        filter_sets shape, then delegates to the shared core. The porosity gate
        (_passes_gate) filters nonporous false positives in live mode."""
        from core.feedback_live_adapter import live_results_to_filter_sets
        filter_sets = live_results_to_filter_sets(live_results)
        self.ingest(filter_sets, iteration, hypothesis_id, hypothesis, parent_id)

    # ------------------------------------------------------------- distillers
    def _update_frontier(self, candidates: List[Dict[str, Any]]) -> None:
        """Merge new candidates into the top-K frontier, dedup by structure."""
        by_struct: Dict[str, Dict[str, Any]] = {}
        for entry in self.frontier + candidates:
            key = entry.get("filename") or entry.get("structure")
            kept = by_struct.get(key)
            if kept is None or self._is_better(entry["value"], kept["value"]):
                by_struct[key] = entry
        merged = sorted(by_struct.values(), key=lambda e: e["value"],
                        reverse=(self.metric_direction == "maximize"))
        self.frontier = merged[:self.frontier_k]

    def _topk_cutoff(self) -> Optional[float]:
        """Value of the worst structure currently kept in the top-K frontier —
        i.e. the bar a new candidate must beat to enter the portfolio."""
        return self.frontier[-1]["value"] if self.frontier else None

    def _topk_median(self) -> Optional[float]:
        """Median value of the current top-K portfolio (typical portfolio quality)."""
        vals = [e["value"] for e in self.frontier]
        return float(pd.Series(vals).median()) if vals else None

    _STOP_TOKENS = {"node", "linker", "with", "and", "the", "based", "nan", "bkbn", "subs"}

    def _chem_tokens(self, structure: str) -> set:
        """Tokenize a blinding-safe structure label into chemistry tokens
        (metals, topology, backbone tags). No database identity leaked."""
        import re
        toks = set()
        for t in re.split(r"[\s,\[\]|:()]+", str(structure)):
            t = t.strip()
            if len(t) >= 2 and t.lower() not in self._STOP_TOKENS:
                toks.add(t)
        return toks

    def _topk_token_set(self, frac: float = 0.4) -> set:
        """Winning-family tokens: present in >= frac of the top-K portfolio."""
        from collections import Counter
        n = len(self.frontier)
        if n < 3:
            return set()
        c: Counter = Counter()
        for e in self.frontier:
            for t in self._chem_tokens(e.get("structure", "")):
                c[t] += 1
        return {t for t, k in c.items() if k >= max(2, round(n * frac))}

    def _detect_outside_promising(self, filter_sets: Dict[str, pd.DataFrame],
                                  reference: Optional[float], winning: set,
                                  min_count: int = 3, frac: float = 0.4) -> Optional[list]:
        """Evidence-graft signal: chemistry that BEATS your CURRENT per-iteration
        quality in the random beam but is NOT in your winning family.

        `reference` = this iteration's Z (full-hypothesis) median = the typical
        quality of the agent's current search. NOTE: we use the per-iter quality,
        NOT the cumulative top-K cutoff — the cumulative cutoff only rises (it
        remembers old wins) and so masks a search that is stuck below what's
        achievable. The per-iter reference fires persistently when the current
        search keeps being beaten (the lock-in case), and goes quiet once the
        agent climbs above the random beam (the healthy case).

        Criteria (this iter): >= min_count random structures beat `reference`,
        sharing chemistry (>= frac of them), differing from the winning family.
        Persistence (>=2 iters) is enforced at render time.
        """
        if reference is None:
            return None
        cutoff = reference
        from collections import Counter
        over_tokens: list = []
        n_over = 0
        # Only the 'total' (random) beam is truly OUTSIDE the portfolio: Z/A/F are
        # hypothesis beams that feed the frontier, so anything good they find is
        # already "winning". Random samples the whole DB independently.
        for beam in ("total",):
            df = filter_sets.get(beam)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty or "target" not in df.columns:
                continue
            for _, row in df.iterrows():
                v = row.get("target")
                if pd.isna(v):
                    continue
                if self._is_better(float(v), cutoff):   # direction-aware "beats cutoff"
                    n_over += 1
                    over_tokens.append(self._chem_tokens(self._struct_label(row)))
        if n_over < min_count:                            # criterion 1: count
            return None
        c: Counter = Counter()
        for s in over_tokens:
            for t in s:
                c[t] += 1
        shared = {t for t, k in c.items() if k >= max(2, round(n_over * frac))}  # criterion 2
        different = sorted(shared - winning)              # criterion 3: differs from winning
        return different or None

    def _topk_chemistry(self) -> str:
        """Chemistry tokens shared by most of the top-K portfolio — the 'winning'
        chemistry the agent should EXPLOIT (concentrate on) to lift the median.

        Tokens are parsed from the blinding-safe structure labels (metals,
        topology, backbone tags), so this leaks no database identity.
        """
        import re
        from collections import Counter
        entries = self.frontier
        n = len(entries)
        if n < 3:
            return ""
        toks: Counter = Counter()
        _stop = {"node", "linker", "with", "and", "the", "based", "nan", "bkbn", "subs"}
        for e in entries:
            seen = set()
            for t in re.split(r"[\s,\[\]|:()]+", str(e.get("structure", ""))):
                t = t.strip()
                if len(t) >= 2 and t.lower() not in _stop and t not in seen:
                    seen.add(t)
                    toks[t] += 1
        # tokens present in >= 40% of the top-K
        common = [f"{t}({c * 100 // n}%)" for t, c in toks.most_common(8)
                  if c >= max(2, round(n * 0.4))]
        return ", ".join(common[:6])

    def _update_geometry_envelope(self) -> None:
        """Recompute geometry min/max/median over the current frontier.

        Descriptors with no real data are dropped: a descriptor whose values have
        zero spread (max == min) carries no information. This covers the all-zero
        case — e.g. QMOF has no surface area / void fraction, stored as 0 — so the
        rendered block shows only the geometry dimensions a given database actually
        carries (QMOF -> Di/Df/density; hMOF/PorMake -> all five). If a DB has no
        real geometry at all, the envelope is empty and the geometry block vanishes
        on its own — no per-database hard-coding."""
        env: Dict[str, List[float]] = {}
        for col in _GEOMETRY_COLS:
            vals = [e["geometry"][col] for e in self.frontier
                    if col in e.get("geometry", {})]
            if len(vals) >= 2:
                s = pd.Series(vals)
                lo, hi, med = float(s.min()), float(s.max()), float(s.median())
                if hi - lo < 1e-9:          # zero spread (incl. all-zero) -> no data
                    continue
                env[col] = [round(lo, 3), round(hi, 3), round(med, 3)]
        self.geometry_envelope = env

    def _geometry_margin(self, lookback: int = 3) -> Optional[float]:
        """Direction-aware mean advantage of the geometry-gated beam (Z, full
        hypothesis) over chemistry-only (A) across the last `lookback` iterations.

        > 0  => gating geometry is IMPROVING results (geometry is a useful lever
                for this target, e.g. volumetric uptake).
        <= 0 => geometry is NOT separating good from bad (the target likely does
                not depend on pore geometry, e.g. an electronic property).

        Data-driven relevance signal (not currently rendered — the geometry block
        is observational; kept available for analysis / optional gating)."""
        diffs = []
        for h in self.history[-lookback:]:
            bm = h.get("beam_medians", {})
            z, a = bm.get("z"), bm.get("a")
            if z is None or a is None:
                continue
            diffs.append((z - a) if self.metric_direction == "maximize" else (a - z))
        return float(sum(diffs) / len(diffs)) if diffs else None

    # ---------------------------------------------------------------- outputs
    def render_for_agent(self) -> str:
        """Compact, blinding-safe knowledge block for Agent 1's context.

        Contains NO database/CSV identity and NO oracle (population) values —
        only what the agent's own hypotheses have produced.
        """
        if self.global_best is None and not self.history:
            return ""

        K = self.frontier_k
        n = len(self.frontier)
        lines = ["=== DESIGN MEMORY (your portfolio so far) ==="]

        # Portfolio block — the objective is a strong TOP-K, not a single best.
        if self.frontier:
            gb = self.global_best
            cutoff = self._topk_cutoff()
            med = self._topk_median()
            lines.append(f"PORTFOLIO — your best {n} distinct structures (goal: a strong top-{K}):")
            lines.append(f"  single best:    {gb['value']:.3g} {self.metric_name}  ({gb['structure']})")
            lines.append(f"  top-{K} cutoff:  {cutoff:.3g}   (a candidate must beat this to ENTER your portfolio)")
            lines.append(f"  top-{K} median:  {med:.3g}   (typical quality of your portfolio)")
            if self.history:
                ne = self.history[-1].get("new_entries", 0)
                lines.append(f"  this iteration: {('+' + str(ne) + ' new entries to top-' + str(K)) if ne else 'no new entries to your top-' + str(K)}")
            lines.append(f"GOAL: lift your WHOLE top-{K} — raise the cutoff AND the median, not just the single best.")
            chem = self._topk_chemistry()
            if chem:
                lines.append(f"WINNING CHEMISTRY (shared by your top-{K}): {chem}")
                lines.append(f"  → To RAISE the median, CONCENTRATE this iteration's hypothesis on this winning")
                lines.append(f"    chemistry — refine WITHIN it (small variations), do NOT switch to a new family.")
                lines.append(f"    A hypothesis whose TYPICAL candidate falls below {cutoff:.3g} wastes the iteration;")
                lines.append(f"    aim for a hypothesis where MOST candidates clear {cutoff:.3g}, not just one.")
            else:
                lines.append("  Concentrate on what works to push MORE candidates into your portfolio")
                lines.append("  (this raises the per-iteration median), rather than switching families each iteration.")

            # Evidence-graft (§8-7): a DIFFERENT family beating your cutoff for 2+ iters
            if len(self.history) >= 2:
                cur = self.history[-1].get("outside_promising")
                prev = self.history[-2].get("outside_promising")
                if cur and prev and (set(cur) & set(prev)):
                    common = ", ".join(sorted(set(cur) & set(prev)))
                    lines.append(f"PROMISING OUTSIDE YOUR PORTFOLIO: chemistry [{common}] appears in the "
                                 f"random beam ABOVE your current per-iteration quality for 2+ iterations, "
                                 f"and you are NOT using it.")
                    lines.append(f"  → A different family is consistently beating your portfolio — a better region exists.")
                    lines.append(f"    GRAFT it: add at least one linker branch featuring [{common}] THIS iteration")
                    lines.append(f"    (keep your winning chemistry too — this is expansion, not abandonment).")

        if n > 1:
            lines.append("TOP STRUCTURES (best-first):")
            for i, e in enumerate(self.frontier[:5], 1):
                lines.append(f"  {i}. {e['value']:.3g} — {e['structure']} (iter {e['iter']})")
            lines.append("  NOTE: each structure's Bkbn/Subs tags CO-OCCUR on ONE linker — that specific")
            lines.append("  COMBINATION is what scored high, not the tags taken individually.")
            lines.append("  → To refine: REPRODUCE a winning combination as a SINGLE linker branch (all its")
            lines.append("    tags together in one branch), then vary ONE component at a time (swap/add/remove")
            lines.append("    a single substituent). Do NOT split the combination into separate single-tag")
            lines.append("    branches — that broadens the match pool but dilutes typical quality. Do NOT pile")
            lines.append("    on extra required tags either — your top structures already match thousands of")
            lines.append("    candidates at this specificity, so stay at that level and vary within it.")

        if self.history:
            bm = self.history[-1]["beam_medians"]
            parts = []
            for beam, label in [("z", "full-hypothesis"), ("a", "chemistry-only"),
                                ("f", "metal-only"), ("total", "random-baseline")]:
                if bm.get(beam) is not None:
                    parts.append(f"{label}={bm[beam]:.3g}")
            if parts:
                lines.append("THIS ITERATION beam medians: " + " | ".join(parts))

        # Geometry anchor (v9): OBSERVATIONAL, not prescriptive. We surface the
        # geometry envelope of the agent's OWN best structures, then let the DATA
        # decide whether geometry matters for THIS target by comparing the
        # geometry-gated beam (Z = full hypothesis) against chemistry-only (A).
        # We never assert geometry "is the gate": for electronic targets (e.g.
        # band gap) it usually is not, and the agent should LEARN that from its own
        # beams rather than from a baked-in (volumetric-uptake) assumption. Only
        # descriptors with real data appear (see _update_geometry_envelope), so the
        # block self-suppresses for databases that carry no geometry.
        if self.geometry_envelope:
            label_map = {"di": "LCD/Di (Å)", "df": "PLD/Df (Å)", "sa": "SA (m²/cm³)",
                         "vf": "void fraction", "density": "density (g/cm³)"}
            geo_parts = []
            for col in _GEOMETRY_COLS:
                if col in self.geometry_envelope:
                    lo, hi, med = self.geometry_envelope[col]
                    geo_parts.append(f"{label_map.get(col, col)} {lo:g}–{hi:g} (med {med:g})")
            if geo_parts:
                lines.append("GEOMETRY OF YOUR BEST STRUCTURES (observational — the geometry ranges your")
                lines.append("current top-10 occupy; your per-iteration beams (full-hypothesis vs")
                lines.append("chemistry-only vs metal-only) already show whether geometry, metal, or")
                lines.append("linker is what separates high from low for this target):")
                lines.append("  " + " | ".join(geo_parts))

        if len(self.history) >= 2:
            prog = [f"i{h['iter']}:{h['topk_cutoff']:.3g}" for h in self.history
                    if h.get("topk_cutoff") is not None]
            if prog:
                lines.append(f"PROGRESS (top-{K} cutoff): " + " → ".join(prog))

        return "\n".join(lines)

    def render_facts_only(self) -> str:
        """PRODUCTION render (ULMEM) — FACTS ONLY: single best, top-K frontier + cutoff/median,
        observational geometry envelope, and per-iter cutoff progress.

        Deliberately OMITS every prescriptive directive that render_for_agent() carries
        ("CONCENTRATE on your winning chemistry / do NOT switch families / raise the median /
        REPRODUCE the winning combination"). In research that exploit guidance HURT the PORMAKE
        rare-peak apps and contradicts the v3.0 prompt's explore-when-stalled philosophy; stripping
        it left the ledger net-positive (7/9 apps, CH4 top-0.1% 5/5). Verdict: memory good,
        exploit-guidance bad. Production uses THIS renderer; render_for_agent is legacy."""
        gb = self.global_best
        if gb is None and not self.frontier:
            return ""
        K = self.frontier_k
        lines = ["=== DESIGN MEMORY (factual record of YOUR best sampled structures so far — no directives) ==="]
        if gb:
            lines.append(f"Best so far: {gb['value']:.4g} {self.metric_name}  "
                         f"({gb.get('structure', 'structure')}, iter {gb.get('iter', '?')})")
        fr = self.frontier or []
        if fr:
            import statistics
            vals = [e["value"] for e in fr]
            cut = (min(vals) if self.metric_direction == "maximize" else max(vals))
            med = statistics.median(vals)
            lines.append(f"Your top-{K} distinct sampled structures: "
                         f"cutoff (worst-of-top-{K}) = {cut:.4g}, median = {med:.4g}.")
            lines.append(f"Top {min(5, len(fr))} (best-first):")
            for i, e in enumerate(fr[:5], 1):
                lines.append(f"  {i}. {e['value']:.4g} — {e.get('structure', 'structure')} (iter {e.get('iter', '?')})")
        env = getattr(self, "geometry_envelope", {}) or {}
        if env:
            lab = {"di": "LCD(di)", "df": "PLD(df)", "sa": "SA", "vf": "VF", "density": "density"}
            parts = [f"{lab.get(c, c)} {v[0]:g}-{v[1]:g}(med {v[2]:g})"
                     for c, v in env.items() if isinstance(v, (list, tuple)) and len(v) == 3]
            if parts:
                lines.append("Geometry ranges of your best structures (observational): " + " | ".join(parts))
        hist = getattr(self, "history", []) or []
        prog = [f"i{h['iter']}:{h['topk_cutoff']:.4g}" for h in hist if h.get("topk_cutoff") is not None]
        if len(prog) >= 2:
            lines.append("Best-of-top-K progress (cutoff per iter): " + " -> ".join(prog))
        return "\n".join(lines)

    def geometry_envelope_text(self) -> str:
        """Achievable geometry ranges (observational) for calibration (D-1)."""
        if not self.geometry_envelope:
            return ""
        lines = ["=== GEOMETRY RANGES OBSERVED IN YOUR BEST STRUCTURES (observational) ==="]
        for col, (lo, hi, med) in self.geometry_envelope.items():
            lines.append(f"  {col.upper()}: {lo}–{hi} (median {med})")
        return "\n".join(lines)

    # --------------------------------------------------------------- persist
    def get_snapshot(self) -> Dict[str, Any]:
        """Full structured snapshot (with provenance) for serialization/audit."""
        return {
            "mode": self.mode,
            "metric": {"name": self.metric_name, "direction": self.metric_direction,
                       "type": self.metric_type},
            "global_best": self.global_best,
            "frontier": self.frontier,
            "geometry_envelope": self.geometry_envelope,
            "history": self.history,
        }

    def serialize(self) -> Dict[str, Any]:
        return self.get_snapshot()

    def deserialize(self, state: Dict[str, Any]) -> None:
        self.global_best = state.get("global_best")
        self.frontier = state.get("frontier", [])
        self.geometry_envelope = state.get("geometry_envelope", {})
        self.history = state.get("history", [])

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            self.deserialize(json.load(f))

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.get_snapshot(), f, indent=2, ensure_ascii=False)


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_memory_ledger():
    """Smoke test the ledger with synthetic markscheme-style filter_sets."""
    import tempfile

    print("\n" + "=" * 60)
    print("MEMORY LEDGER MODULE TEST")
    print("=" * 60 + "\n")

    test_dir = tempfile.mkdtemp(prefix="ledger_test_")
    ledger = MemoryLedger(test_dir, metric_name="CH4 Uptake", metric_direction="maximize")

    def _fs(z_vals, a_vals, f_vals, total_vals):
        def mk(vals, tag):
            return pd.DataFrame({
                "filename": [f"pcu+N{tag}{i}+E{i}" for i in range(len(vals))],
                "target": vals,
                "di": [12.0] * len(vals), "df": [8.0] * len(vals),
                "sa": [2300.0] * len(vals), "vf": [0.5] * len(vals),
                "density": [0.5] * len(vals),
            })
        return {"z": mk(z_vals, "Z"), "a": mk(a_vals, "A"),
                "f": mk(f_vals, "F"), "total": mk(total_vals, "T")}

    # iter 1: best 120 (Z); iter 2: best 145 (Z, new best); iter 3: best 132 (regression)
    ledger.ingest_markscheme(_fs([120, 110], [115, 100], [90, 80], [200, 195]), 1, "iter1")
    ledger.ingest_markscheme(_fs([145, 130], [140, 120], [100, 95], [205, 190]), 2, "iter2", parent_id="iter1")
    ledger.ingest_markscheme(_fs([132, 125], [128, 118], [99, 90], [210, 188]), 3, "iter3", parent_id="iter2")
    ledger.save()

    print("\n--- render_for_agent() ---")
    print(ledger.render_for_agent())

    print("\n--- Validation ---")
    assert ledger.global_best["value"] == 145.0, "global_best should be 145 (from Z/A/F)"
    assert ledger.global_best["iter"] == 2, "global_best should be iter 2"
    # Oracle safety: 'total' beam had 200+ values but must NOT leak into global_best
    assert ledger.global_best["value"] < 200, "ORACLE LEAK: total beam must not feed global_best"
    assert os.path.exists(ledger.path), "ledger json should be saved"
    print("✓ global_best = 145 @ iter2 (Z/A/F only, total=200+ excluded)")
    print("✓ MEMORY LEDGER TEST PASSED")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_memory_ledger()
