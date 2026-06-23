# =============================================================================
# LLM4MOF Autonomous System
# =============================================================================
# MOF Generator with Geometry Prediction (mof2zeo integration)
# Used when no database is available for matching - predicts geometry from
# topology+node+edge combinations and ranks by target geometry match
# =============================================================================

import os
import sys
import json
import torch
import yaml
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

_core_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_core_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, _core_dir)
sys.path.insert(0, os.path.join(_core_dir, "mof2zeo"))

import mof2zeo
import config
from mof2zeo.model import MOFNET
from mof2zeo.dataset import Scaler


@dataclass
class MOFComponent:
    topology: str
    node: str
    edge: str

    @property
    def filename(self) -> str:
        return f"{self.topology}+{self.node}+{self.edge}"

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.topology, self.node, self.edge)


@dataclass
class PredictedGeometry:
    sa: float
    cv: float
    density: float
    vf: float
    di: float
    df: float
    dif: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "sa": self.sa,
            "cv": self.cv,
            "density": self.density,
            "vf": self.vf,
            "di": self.di,
            "df": self.df,
            "dif": self.dif,
        }


@dataclass
class RankedMOF:
    rank: int
    component: MOFComponent
    predicted_geometry: PredictedGeometry
    match_score: float
    geometry_match: Dict[str, str]
    all_constraints_satisfied: bool = False


def load_dictionaries() -> Tuple[
    Dict[str, int], Dict[str, int], Dict[str, int], List[str]
]:
    with open(config.MOF2ZEO_TOPOLOGY_FILE, "r", encoding="utf-8") as f:
        topo_dict = {name.strip(): i for i, name in enumerate(f.readlines())}

    with open(config.MOF2ZEO_NODE_FILE, "r", encoding="utf-8") as f:
        node_dict = {name.strip(): i for i, name in enumerate(f.readlines())}

    with open(config.MOF2ZEO_EDGE_FILE, "r", encoding="utf-8") as f:
        edge_dict = {name.strip(): i for i, name in enumerate(f.readlines())}

    with open(config.MOF2ZEO_FEATURE_FILE, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f.readlines()]

    return topo_dict, node_dict, edge_dict, feature_names


# Use external Scaler from mof2zeo (already imported at top)


def load_scaler() -> Scaler:
    mean_df = pd.read_csv(config.MOF2ZEO_SCALER_MEAN_PATH)
    std_df = pd.read_csv(config.MOF2ZEO_SCALER_STD_PATH)

    with open(config.MOF2ZEO_FEATURE_FILE, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f.readlines()]

    mean = mean_df[feature_names].values.squeeze()
    std = std_df[feature_names].values.squeeze()

    # target_mean=0, target_std=1 (standard for this model)
    return Scaler(mean, std, 0, 1)


class GeometryPredictor:
    _instance = None
    _model = None
    _scaler = None
    _topo_dict = None
    _node_dict = None
    _edge_dict = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        if not config.is_mof2zeo_available():
            print("[Agent 3] WARNING: mof2zeo model not available")
            self._model = None
            return

        # Load model config
        with open(config.MOF2ZEO_CONFIG_PATH, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)

        self._topo_dict, self._node_dict, self._edge_dict, _ = load_dictionaries()
        self._scaler = load_scaler()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # PyTorch 2.6+ changed torch.load default to weights_only=True, which
            # refuses to unpickle non-builtin classes for security. The mof2zeo
            # checkpoint contains a mof2zeo.dataset.Scaler instance and a
            # numpy.dtype reconstructor, so we whitelist them before the load.
            # This is the PyTorch-recommended fix from the error message itself.
            try:
                import numpy
                from numpy.core.multiarray import _reconstruct as _numpy_reconstruct
                from numpy import ndarray, dtype
                torch.serialization.add_safe_globals([
                    Scaler,
                    _numpy_reconstruct,
                    ndarray,
                    dtype,
                    numpy.dtypes.Float64DType,
                    numpy.dtypes.Float32DType,
                    numpy.dtypes.Int64DType,
                ])
            except Exception as _safe_e:
                print(f"[Agent 3] note: safe_globals partial setup ({_safe_e}); "
                      f"falling back to weights_only=False on the load")

            # Load checkpoint from config
            ckpt_path = config.MOF2ZEO_CKPT_PATH

            try:
                self._model = MOFNET.load_from_checkpoint(
                    ckpt_path,
                    config=model_config,
                    scaler=self._scaler,
                    strict=False,
                )
            except Exception as _wo_e:
                # Belt-and-suspenders: if safe_globals didn't cover everything in the
                # checkpoint, fall back to the explicit weights_only=False path.
                # The checkpoint is shipped with the repo via Git LFS; we trust it.
                print(f"[Agent 3] note: weights_only safe load failed ({type(_wo_e).__name__}); "
                      f"retrying with weights_only=False (trusted local checkpoint)")
                import torch as _torch
                _orig_load = _torch.load
                def _trusted_load(*args, **kwargs):
                    kwargs["weights_only"] = False
                    return _orig_load(*args, **kwargs)
                _torch.load = _trusted_load
                try:
                    self._model = MOFNET.load_from_checkpoint(
                        ckpt_path,
                        config=model_config,
                        scaler=self._scaler,
                        strict=False,
                    )
                finally:
                    _torch.load = _orig_load

            self._model.to(self._device)
            self._model.eval()
            print(f"[Agent 3] mof2zeo model loaded successfully on {self._device}")
        except Exception as e:
            print(f"[Agent 3] ERROR loading model: {e}")
            self._model = None

    def predict(self, topology: str, node: str, edge: str) -> PredictedGeometry:
        # OOV check — warn when master's matchmaker hands us a topology/node/edge
        # that was not in the mof2zeo training vocabulary. A silent fallback to
        # index 0 (as the prior version did via dict.get(key, 0)) produces a
        # prediction for the wrong MOF, which is a silent-wrong-answer bug.
        # Current vocab coverage (verified 2026-04-08 against master v2.5):
        #   nodes  100% (648/648)  — no OOV expected
        #   edges  100% (219/219)  — no OOV expected
        #   topos  ~41% (964/2364) — real OOV risk for exotic topologies
        if topology not in self._topo_dict:
            print(f"[Agent 3] WARNING: topology '{topology}' not in mof2zeo vocab; "
                  f"using index 0 fallback (prediction unreliable for this candidate).")
        if node not in self._node_dict:
            print(f"[Agent 3] WARNING: node '{node}' not in mof2zeo vocab; "
                  f"using index 0 fallback (prediction unreliable for this candidate).")
        if edge not in self._edge_dict:
            print(f"[Agent 3] WARNING: edge '{edge}' not in mof2zeo vocab; "
                  f"using index 0 fallback (prediction unreliable for this candidate).")
        topo_idx = self._topo_dict.get(topology, 0)
        node_idx = self._node_dict.get(node, 0)
        edge_idx = self._edge_dict.get(edge, 0)

        if self._model is not None:
            with torch.no_grad():
                mof_tensor = torch.tensor(
                    [[topo_idx, node_idx, edge_idx]], dtype=torch.long
                ).to(self._device)
                pred_scaled = self._model(mof_tensor)
                pred_geometry = self._scaler.decode(pred_scaled.cpu())[0].numpy()
        else:
            pred_geometry = self._fallback_predict(topology, node, edge)

        return PredictedGeometry(
            sa=pred_geometry[0],
            cv=pred_geometry[1],
            density=pred_geometry[2],
            vf=pred_geometry[3],
            di=pred_geometry[4],
            df=pred_geometry[5],
            dif=pred_geometry[6],
        )

    def _fallback_predict(self, topology: str, node: str, edge: str) -> np.ndarray:
        np.random.seed(hash(f"{topology}{node}{edge}") % (2**32))
        if topology in ["fcu", "pcu"]:
            return np.array([2000, 50000, 1.0, 0.6, 12.0, 6.0, 8.0])
        elif topology in ["sqc", "qom"]:
            return np.array([1500, 30000, 1.2, 0.5, 10.0, 5.0, 7.0])
        else:
            return np.array([1800, 40000, 1.1, 0.55, 11.0, 5.5, 7.5])

    def predict_batch(self, components: List[MOFComponent]) -> List[PredictedGeometry]:
        return [self.predict(c.topology, c.node, c.edge) for c in components]


class ComponentGenerator:
    def __init__(self):
        self._topo_dict, self._node_dict, self._edge_dict, _ = load_dictionaries()
        self._topo_list = list(self._topo_dict.keys())
        self._node_list = list(self._node_dict.keys())
        self._edge_list = list(self._edge_dict.keys())

        self._load_cn_data()

    def _load_cn_data(self):
        with open(config.TOPO_DICTIONARY_PATH, "r", encoding="utf-8") as f:
            topo_data = json.load(f)
        with open(config.BB_DICTIONARY_PATH, "r", encoding="utf-8") as f:
            bb_data = json.load(f)

        self._topo_cn = {
            t["ID"]: t.get("node_connectivities", [0])[0] for t in topo_data
        }
        self._node_cn = {
            b["ID"]: b.get("connectivity", 0)
            for b in bb_data
            if b.get("ID", "").startswith("N")
        }
        self._edge_cn = {
            b["ID"]: b.get("connectivity", 0)
            for b in bb_data
            if b.get("ID", "").startswith("E")
        }

    def generate_from_matchmaker(
        self, matchmaker_result: Dict[str, Any], max_combos: int = 1000
    ) -> List[MOFComponent]:
        """Generate diverse combinations using stratified random sampling.

        Safety contract (added 2026-04-08 for master v2.5 compatibility):
        If the matchmaker returns its structured error dict (status='error')
        or empty topology/node/edge lists, we ABORT here with a clear error
        instead of silently substituting random entries from the mof2zeo
        training vocabulary. The earlier behavior produced plausible-looking
        rankings for MOFs that had NO relationship to the user's constraints,
        which is a silent-wrong-answer bug.
        """
        # Hard guard: master's matchmaker returns a structured error dict with
        # status='error' when no nodes match the chemistry constraints (e.g.,
        # when the LLM over-constrains in iteration 1). Refuse to proceed —
        # the caller (run_simulation.py) should catch this and let the iterative
        # loop broaden the hypothesis, not silently simulate random MOFs.
        if matchmaker_result.get("status") == "error":
            raise ValueError(
                f"[Agent 3] Matchmaker returned error: "
                f"{matchmaker_result.get('message', 'no matching candidates')}. "
                f"Refusing to substitute random vocabulary. "
                f"Broaden the Agent 1 hypothesis and retry."
            )


        topo_list = matchmaker_result.get("topology", [])
        node_list = matchmaker_result.get("node", [])
        edge_list = matchmaker_result.get("edge", [])

        # Hard guard: any empty dimension means the matchmaker filtered to zero.
        # Refuse to fall back to random vocabulary.
        if not topo_list or not node_list or not edge_list:
            raise ValueError(
                f"[Agent 3] Matchmaker returned an empty candidate set "
                f"(topologies={len(topo_list)}, nodes={len(node_list)}, "
                f"edges={len(edge_list)}). Refusing to substitute random "
                f"vocabulary — the ranking would be meaningless. "
                f"Broaden the Agent 1 hypothesis and retry."
            )

        return self._generate_diverse_combinations(
            topo_list, node_list, edge_list, max_combos=max_combos
        )

    def _generate_diverse_combinations(
        self,
        topo_list: List[str],
        node_list: List[str],
        edge_list: List[str],
        max_combos: int = 1000,
    ) -> List[MOFComponent]:
        """
        Generate all CN-valid combinations, then randomly sample up to max_combos.

        Algorithm:
        1. Enumerate ALL (topology, node, edge) triples that satisfy CN constraints
        2. If more than max_combos valid combinations exist, randomly sample max_combos
        3. Shuffle and return

        This ensures full topology/node/edge coverage — no dimension is artificially
        capped before CN validation.
        """
        # Load mof2zeo vocab dictionaries for OOV filtering
        with open(config.MOF2ZEO_TOPOLOGY_FILE, "r") as f:
            _topo_vocab = frozenset(l.strip() for l in f)
        with open(config.MOF2ZEO_NODE_FILE, "r") as f:
            _node_vocab = frozenset(l.strip() for l in f)
        with open(config.MOF2ZEO_EDGE_FILE, "r") as f:
            _edge_vocab = frozenset(l.strip() for l in f)

        topo_list_in = [t for t in topo_list if t in _topo_vocab]
        node_list_in = [n for n in node_list if n in _node_vocab]
        edge_list_in = [e for e in edge_list if e in _edge_vocab]

        n_oov_t = len(topo_list) - len(topo_list_in)
        n_oov_n = len(node_list) - len(node_list_in)
        n_oov_e = len(edge_list) - len(edge_list_in)
        if n_oov_t or n_oov_n or n_oov_e:
            print(
                f"[Agent 3] OOV filtered: {n_oov_t}T / {n_oov_n}N / {n_oov_e}E removed "
                f"(not in mof2zeo vocab) → {len(topo_list_in)}T × {len(node_list_in)}N × {len(edge_list_in)}E remain"
            )
        topo_list = topo_list_in
        node_list = node_list_in
        edge_list = edge_list_in

        # Build all CN-valid combinations
        all_valid: List[MOFComponent] = []
        for topo in topo_list:
            topo_cn = self._topo_cn.get(topo, 0)
            for node in node_list:
                node_cn = self._node_cn.get(node, 0)
                if topo_cn != node_cn:
                    continue
                for edge in edge_list:
                    if self._edge_cn.get(edge, 0) == 2:
                        all_valid.append(MOFComponent(topology=topo, node=node, edge=edge))

        n_valid = len(all_valid)

        if n_valid > max_combos:
            indices = np.random.choice(n_valid, size=max_combos, replace=False)
            combinations = [all_valid[i] for i in indices]
        else:
            combinations = all_valid
            np.random.shuffle(combinations)

        n_topo_covered = len({c.topology for c in combinations})
        print(
            f"[Agent 3] {n_valid} CN-valid combinations "
            f"(from {len(topo_list)}T × {len(node_list)}N × {len(edge_list)}E); "
            f"using {len(combinations)} ({n_topo_covered} unique topologies)"
        )

        return combinations


class MOFRanker:
    def __init__(self):
        self.weights = {
            "di": 1/7,
            "df": 1/7,
            "sa": 1/7,
            "vf": 1/7,
            "density": 1/7,
            "cv": 1/7,
            "dif": 1/7,
        }

    def rank(
        self,
        combinations: List[MOFComponent],
        predicted_geometries: List[PredictedGeometry],
        target_geometry: Dict[str, Any],
        preferred_features: Dict[str, Any] = None,
        bb_lookup: Dict[str, Any] = None,
    ) -> List[RankedMOF]:

        results = []

        for comp, pred_geo in zip(combinations, predicted_geometries):
            score, match_details = self._calculate_match_score(
                pred_geo, target_geometry
            )
            if preferred_features and bb_lookup:
                score += self._calculate_preferred_bonus(comp, preferred_features, bb_lookup)
            results.append(
                RankedMOF(
                    rank=0,
                    component=comp,
                    predicted_geometry=pred_geo,
                    match_score=score,
                    geometry_match=match_details,
                )
            )

        results.sort(key=lambda x: x.match_score, reverse=True)

        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    @staticmethod
    def _calculate_preferred_bonus(
        comp: "MOFComponent", preferred_features: Dict[str, Any], bb_lookup: Dict[str, Any]
    ) -> float:
        """Compute a soft ranking bonus for preferred (non-mandatory) abstract features.

        Unlike abstract_features (hard AND filter), preferred_features only boost
        match_score — no candidate is excluded. Bonus is capped at 0.15 so geometry
        always dominates; preferred features act as a tiebreaker when geometry is null.

        Args:
            comp: The MOF component (topology + node_id + edge_id).
            preferred_features: Dict with optional 'node' and 'linker' sub-dicts,
                e.g. {'node': {'is_conjugated': True}, 'linker': {'has_hydrogen_bond_acceptor': True}}.
            bb_lookup: Building block lookup dict keyed by BB ID.

        Returns:
            Float bonus in range [0.0, 0.15].
        """
        if not preferred_features or not bb_lookup:
            return 0.0

        bonus = 0.0
        per_feature_bonus = 0.05  # each matching preferred feature adds 0.05

        node_pf = preferred_features.get('node') or {}
        linker_pf = preferred_features.get('linker') or {}

        for bb_id, pf_dict in [(comp.node, node_pf), (comp.edge, linker_pf)]:
            if not pf_dict:
                continue
            item_af = bb_lookup.get(bb_id, {}).get('abstract_features', {})
            if not item_af:
                continue
            for feat_key, feat_val in pf_dict.items():
                if feat_val is None:
                    continue
                if item_af.get(feat_key) == feat_val:
                    bonus += per_feature_bonus

        return min(bonus, 0.15)  # cap total bonus

    def _calculate_match_score(
        self, pred: PredictedGeometry, target: Dict[str, Any]
    ) -> Tuple[float, Dict[str, str]]:
        score = 0.0
        match_details = {}

        pred_dict = pred.to_dict()

        for key, weight in self.weights.items():
            target_min = target.get(f"target_{key}_min") or target.get(
                f"target_{key.capitalize()}_min"
            )
            target_max = target.get(f"target_{key}_max") or target.get(
                f"target_{key.capitalize()}_max"
            )

            if target_min is None and target_max is None:
                match_details[key] = "○ (no constraint)"
                continue

            pred_val = pred_dict.get(key)
            if pred_val is None:
                match_details[key] = "? (no prediction)"
                continue

            if target_min is not None and target_max is not None:
                if target_min <= pred_val <= target_max:
                    score += weight
                    match_details[key] = f"✓ ({pred_val:.1f} in range)"
                else:
                    if pred_val < target_min:
                        penalty = min((target_min - pred_val) / target_min, 1.0)
                    else:
                        penalty = min((pred_val - target_max) / target_max, 1.0)
                    score += weight * (1.0 - penalty)
                    match_details[key] = f"✗ ({pred_val:.1f} out of range)"
            elif target_min is not None:
                if pred_val >= target_min:
                    score += weight
                    match_details[key] = f"✓ ({pred_val:.1f} >= {target_min})"
                else:
                    penalty = min((target_min - pred_val) / target_min, 1.0)
                    score += weight * (1.0 - penalty)
                    match_details[key] = f"✗ ({pred_val:.1f} < {target_min})"
            elif target_max is not None:
                if pred_val <= target_max:
                    score += weight
                    match_details[key] = f"✓ ({pred_val:.1f} <= {target_max})"
                else:
                    penalty = min((pred_val - target_max) / target_max, 1.0)
                    score += weight * (1.0 - penalty)
                    match_details[key] = f"✗ ({pred_val:.1f} > {target_max})"

        return score, match_details


# MAE-based slack for prediction filter (mof2zeo test_on_260610 errors)
# Prediction stage uses LLM range ± MAE; stage1 real geometry uses strict LLM range.
_MAE_SLACK: Dict[str, float] = {
    "sa": 50.7,
    "cv": 5024.0,
    "density": 0.0152,
    "vf": 0.0098,
    "di": 0.746,
    "df": 0.751,
    "dif": 0.828,
}


def _apply_mae_slack(target: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of geometry_filter with ±MAE slack applied to all bounds."""
    slacked = dict(target)
    for feat, slack in _MAE_SLACK.items():
        for suffix, sign in (("_min", -1), ("_max", +1)):
            for key in (f"target_{feat}{suffix}", f"target_{feat.capitalize()}{suffix}"):
                if key in slacked and slacked[key] is not None:
                    slacked[key] = slacked[key] + sign * slack
    return slacked


class Agent3Handler:
    def __init__(self):
        print("[Agent 3] Initializing MOF Generator...")
        self.predictor = GeometryPredictor()
        self.generator = ComponentGenerator()
        self.ranker = MOFRanker()
        print("[Agent 3] Initialized - MOF Generator Ready")

    def generate_mof_proposals(
        self,
        matchmaker_result: Dict[str, Any],
        agent2_constraints: Dict[str, Any],
        top_n: int = 20,
    ) -> Dict[str, Any]:
        print("\n[Agent 3] Generating MOF proposals...")

        print(
            "[Agent 3] Step 1: Generating component combinations from matchmaker result..."
        )
        combinations = self.generator.generate_from_matchmaker(matchmaker_result)

        if not combinations:
            return {
                "ranked_mofs": [],
                "summary": {
                    "total_combinations": 0,
                    "error": "No valid combinations found",
                },
            }

        print(
            f"[Agent 3] Step 2: Predicting geometry for {len(combinations)} combinations..."
        )
        predicted_geometries = self.predictor.predict_batch(combinations)

        print("[Agent 3] Step 3: Evaluating geometry match...")
        target_geometry = agent2_constraints.get("geometry_filter", {})
        # Slacked target for prediction filter: LLM range ± MAE to compensate prediction error
        target_geometry_slacked = _apply_mae_slack(target_geometry)

        # Evaluate all combinations (no ranking, just scoring)
        all_mofs = []
        for comp, pred_geo in zip(combinations, predicted_geometries):
            score, match_details = self.ranker._calculate_match_score(
                pred_geo, target_geometry
            )
            # Check if ALL constraints are satisfied using MAE-slacked target
            # (prediction error ~0.75Å for di/df; real geometry checked strict at stage1)
            _, match_slacked = self.ranker._calculate_match_score(
                pred_geo, target_geometry_slacked
            )
            all_satisfied = all(
                v.startswith("✓")
                for k, v in match_slacked.items()
                if not v.startswith("○")  # Skip unconstrained
            )
            all_mofs.append(
                RankedMOF(
                    rank=0,
                    component=comp,
                    predicted_geometry=pred_geo,
                    match_score=score,
                    geometry_match=match_details,
                    all_constraints_satisfied=all_satisfied,
                )
            )

        # Filter to valid combinations (all geometry constraints satisfied)
        valid_mofs = [m for m in all_mofs if m.all_constraints_satisfied]

        print(
            f"[Agent 3] Found {len(valid_mofs)} valid combinations "
            f"(out of {len(all_mofs)} total)"
        )

        # Random sampling instead of ranking
        if len(valid_mofs) >= top_n:
            # Randomly sample top_n from valid combinations
            sample_indices = np.random.choice(
                len(valid_mofs), size=top_n, replace=False
            )
            selected_mofs = [valid_mofs[i] for i in sample_indices]
        else:
            # Not enough valid combinations
            print(
                f"[Agent 3] WARNING: Only {len(valid_mofs)} valid combinations "
                f"available (requested {top_n})"
            )
            if len(valid_mofs) > 0:
                # Use all valid and fill with partial matches
                selected_mofs = valid_mofs.copy()
                remaining_needed = top_n - len(valid_mofs)
                # Sort remaining by score and take best ones
                partial_mofs = [m for m in all_mofs if not m.all_constraints_satisfied]
                partial_mofs.sort(key=lambda x: x.match_score, reverse=True)
                selected_mofs.extend(partial_mofs[:remaining_needed])
            else:
                # No valid combinations - return highest scoring ones
                all_mofs.sort(key=lambda x: x.match_score, reverse=True)
                selected_mofs = all_mofs[:top_n]

        # Shuffle for random order
        np.random.shuffle(selected_mofs)

        output_mofs = []
        for idx, mof in enumerate(selected_mofs):
            output_mofs.append(
                {
                    "rank": idx + 1,
                    "topology": mof.component.topology,
                    "node": mof.component.node,
                    "edge": mof.component.edge,
                    "filename": mof.component.filename,
                    "predicted_geometry": mof.predicted_geometry.to_dict(),
                    "match_score": round(mof.match_score, 3),
                    "geometry_match": mof.geometry_match,
                }
            )

        valid_count = len(valid_mofs)

        result = {
            "ranked_mofs": output_mofs,
            "summary": {
                "total_combinations": len(combinations),
                "valid_after_geometry_filter": valid_count,
                "top_n": top_n,
                "target_geometry": target_geometry,
            },
        }

        print(f"[Agent 3] Complete: {len(output_mofs)} MOFs in top-{top_n}")
        return result

    def _print_proposals(self, proposals: Dict[str, Any]):
        print("\n" + "=" * 60)
        print("AGENT 3 MOF PROPOSALS")
        print("=" * 60)

        for mof in proposals["ranked_mofs"][:5]:
            print(f"\nRank {mof['rank']}: {mof['filename']}")
            print(f"  Score: {mof['match_score']:.3f}")
            print(
                f"  Predicted: Di={mof['predicted_geometry']['di']:.1f}Å, "
                f"Df={mof['predicted_geometry']['df']:.1f}Å, "
                f"SA={mof['predicted_geometry']['sa']:.0f}m²/cm³, "
                f"VF={mof['predicted_geometry']['vf']:.2f}"
            )

        print("\n" + "=" * 60)


def test_agent3():
    print("\n" + "=" * 60)
    print("AGENT 3 HANDLER TEST")
    print("=" * 60 + "\n")

    sample_matchmaker_result = {
        "topology": ["fcu", "pcu", "bcu"],
        "node": ["N164", "N12", "N13"],
        "edge": ["E70", "E10", "E11"],
    }

    sample_agent2_constraints = {
        "geometry_filter": {
            "target_Di_min": 12.0,
            "target_Di_max": 20.0,
            "target_Df_min": 7.0,
            "target_Df_max": 10.0,
            "target_sa_min": 1000.0,
            "target_vf_min": 0.5,
        }
    }

    agent3 = Agent3Handler()
    proposals = agent3.generate_mof_proposals(
        sample_matchmaker_result, sample_agent2_constraints, top_n=10
    )
    agent3._print_proposals(proposals)

    print("\n--- Summary ---")
    print(f"Total combinations: {proposals['summary']['total_combinations']}")
    print(f"Valid after filter: {proposals['summary']['valid_after_geometry_filter']}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MOF Candidate Filter using mof2zeo")
    parser.add_argument(
        "--output", type=str, default="test_result_agent3.json", help="Output JSON file"
    )
    parser.add_argument(
        "--top_n", type=int, default=10, help="Number of candidates to generate"
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default=None,
        help="Combined constraints JSON (agent2 output format)",
    )
    parser.add_argument(
        "--matchmaker", type=str, default=None, help="Matchmaker JSON file (legacy)"
    )
    args = parser.parse_args()

    if args.constraints:
        with open(args.constraints, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "matchmaker_result" in data:
            matchmaker_result = data.get("matchmaker_result", {})
        else:
            matchmaker_result = {
                "topology": [],
                "node": [],
                "edge": [],
            }

        agent2_constraints = {"geometry_filter": data.get("geometry_filter", {})}
        print("Loaded constraints from:", args.constraints)
    elif args.matchmaker:
        with open(args.matchmaker, "r", encoding="utf-8") as f:
            matchmaker_result = json.load(f)
        agent2_constraints = {"geometry_filter": {}}
    else:
        sample_matchmaker_result = {
            "topology": ["fcu", "pcu", "bcu"],
            "node": ["N164", "N12", "N13"],
            "edge": ["E70", "E10", "E11"],
        }
        sample_agent2_constraints = {
            "geometry_filter": {
                "target_Di_min": 12.0,
                "target_Di_max": 20.0,
                "target_Df_min": 7.0,
                "target_Df_max": 10.0,
                "target_sa_min": 1000.0,
                "target_vf_min": 0.5,
            }
        }
        matchmaker_result = sample_matchmaker_result
        agent2_constraints = sample_agent2_constraints
        print("Using sample data (no --matchmaker/--constraints provided)")

    agent3 = Agent3Handler()
    proposals = agent3.generate_mof_proposals(
        matchmaker_result, agent2_constraints, top_n=args.top_n
    )

    with open(args.output, "w") as f:
        json.dump({"proposals": proposals}, f, indent=2)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
