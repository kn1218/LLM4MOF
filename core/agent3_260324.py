# =============================================================================
# LLM2POR Autonomous System - Agent 3 Handler
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.mof2zeo.model import MOFNET


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


def load_dictionaries() -> Tuple[
    Dict[str, int], Dict[str, int], Dict[str, int], List[str]
]:
    with open(config.MOF2ZEO_TOPOLOGY_FILE, "r") as f:
        topo_dict = {name.strip(): i for i, name in enumerate(f.readlines())}

    with open(config.MOF2ZEO_NODE_FILE, "r") as f:
        node_dict = {name.strip(): i for i, name in enumerate(f.readlines())}

    with open(config.MOF2ZEO_EDGE_FILE, "r") as f:
        edge_dict = {name.strip(): i for i, name in enumerate(f.readlines())}

    with open(config.MOF2ZEO_FEATURE_FILE, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    return topo_dict, node_dict, edge_dict, feature_names


class Scaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.eps = 1e-6

    def decode(self, batch: torch.Tensor) -> torch.Tensor:
        return batch * self.std + self.mean


def load_scaler() -> Scaler:
    mean_df = pd.read_csv(config.MOF2ZEO_SCALER_MEAN_PATH)
    std_df = pd.read_csv(config.MOF2ZEO_SCALER_STD_PATH)

    with open(config.MOF2ZEO_FEATURE_FILE, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    mean = mean_df[feature_names].values.squeeze()
    std = std_df[feature_names].values.squeeze()

    return Scaler(mean, std)


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

        model_config = {
            "latent_dim": config.MOF2ZEO_LATENT_DIM,
            "hid_dim1": config.MOF2ZEO_HID_DIM1,
            "hid_dim2": config.MOF2ZEO_HID_DIM2,
            "desc_dim": config.MOF2ZEO_DESC_DIM,
            "exp_name": "mof2zeo",
        }

        self._topo_dict, self._node_dict, self._edge_dict, _ = load_dictionaries()
        self._scaler = load_scaler()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self._model = MOFNET.load_from_checkpoint(
                config.MOF2ZEO_CKPT_PATH,
                config=model_config,
                scaler=self._scaler,
                strict=False,
            )
            self._model.to(self._device)
            self._model.eval()
            print(f"[Agent 3] mof2zeo model loaded on {self._device}")
        except Exception as e:
            print(f"[Agent 3] ERROR loading model: {e}")
            self._model = None

    def predict(self, topology: str, node: str, edge: str) -> PredictedGeometry:
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

    def generate_from_matchmaker(
        self, matchmaker_result: Dict[str, Any]
    ) -> List[MOFComponent]:
        topo_list = matchmaker_result.get("topology", [])
        node_list = matchmaker_result.get("node", [])
        edge_list = matchmaker_result.get("edge", [])

        if not topo_list:
            topo_list = self._topo_list[:50]
        if not node_list:
            node_list = self._node_list[:100]
        if not edge_list:
            edge_list = self._edge_list[:50]

        combinations = []
        max_combos = 1000

        for topo in topo_list:
            for node in node_list:
                for edge in edge_list:
                    combinations.append(
                        MOFComponent(topology=topo, node=node, edge=edge)
                    )
                    if len(combinations) >= max_combos:
                        break
                if len(combinations) >= max_combos:
                    break
            if len(combinations) >= max_combos:
                break

        print(
            f"[Agent 3] Generated {len(combinations)} combinations from matchmaker result"
        )
        return combinations


class MOFRanker:
    def __init__(self):
        self.weights = {
            "di": 0.20,
            "df": 0.20,
            "sa": 0.20,
            "vf": 0.20,
            "density": 0.10,
            "cv": 0.05,
            "dif": 0.05,
        }

    def rank(
        self,
        combinations: List[MOFComponent],
        predicted_geometries: List[PredictedGeometry],
        target_geometry: Dict[str, Any],
    ) -> List[RankedMOF]:

        results = []

        for comp, pred_geo in zip(combinations, predicted_geometries):
            score, match_details = self._calculate_match_score(
                pred_geo, target_geometry
            )
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

    def _calculate_match_score(
        self, pred: PredictedGeometry, target: Dict[str, Any]
    ) -> Tuple[float, Dict[str, str]]:
        score = 0.0
        match_details = {}

        pred_dict = pred.to_dict()

        for key, weight in self.weights.items():
            target_min = target.get(f"target_{key}_min")
            target_max = target.get(f"target_{key}_max")

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

        print("[Agent 3] Step 3: Ranking by geometry match...")
        target_geometry = agent2_constraints.get("geometry_filter", {})
        ranked_mofs = self.ranker.rank(
            combinations, predicted_geometries, target_geometry
        )

        top_mofs = ranked_mofs[:top_n]

        output_mofs = []
        for mof in top_mofs:
            output_mofs.append(
                {
                    "rank": mof.rank,
                    "topology": mof.component.topology,
                    "node": mof.component.node,
                    "edge": mof.component.edge,
                    "filename": mof.component.filename,
                    "predicted_geometry": mof.predicted_geometry.to_dict(),
                    "match_score": round(mof.match_score, 3),
                    "geometry_match": mof.geometry_match,
                }
            )

        valid_count = sum(1 for m in ranked_mofs if m.match_score > 0.5)

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
                f"SA={mof['predicted_geometry']['sa']:.0f}m²/g, "
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


if __name__ == "__main__":
    test_agent3()
