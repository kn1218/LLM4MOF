# =============================================================================
# LLM2POR Autonomous System
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
import pandas as pd
import numpy as np
import yaml


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
    with open(config.MOF2ZEO_TOPOLOGY_FILE, "r") as f:
        topo_dict = {name.strip(): i for i, name in enumerate(f.readlines())}

    with open(config.MOF2ZEO_NODE_FILE, "r") as f:
        node_dict = {name.strip(): i for i, name in enumerate(f.readlines())}

    with open(config.MOF2ZEO_EDGE_FILE, "r") as f:
        edge_dict = {name.strip(): i for i, name in enumerate(f.readlines())}

    with open(config.MOF2ZEO_FEATURE_FILE, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    return topo_dict, node_dict, edge_dict, feature_names


# Use external Scaler from mof2zeo (already imported at top)


def load_scaler() -> Scaler:
    mean_df = pd.read_csv(config.MOF2ZEO_SCALER_MEAN_PATH)
    std_df = pd.read_csv(config.MOF2ZEO_SCALER_STD_PATH)

    with open(config.MOF2ZEO_FEATURE_FILE, "r") as f:
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

        # Load original config from mof2zeo
        with open(
            "/home/users/seunghh/_hd1/autollm/mof2zeo/mof2zeo/config.yaml", "r"
        ) as f:
            model_config = yaml.safe_load(f)

        self._topo_dict, self._node_dict, self._edge_dict, _ = load_dictionaries()
        self._scaler = load_scaler()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Load checkpoint from original path
            ckpt_path = "/home/users/seunghh/_hd1/autollm/260225_mof2zeo/ckpt/epoch=478-step=213634.ckpt"

            self._model = MOFNET.load_from_checkpoint(
                ckpt_path,
                config=model_config,
                scaler=self._scaler,
                strict=False,
            )

            self._model.to(self._device)
            self._model.eval()
            print(f"[Agent 3] mof2zeo model loaded successfully on {self._device}")
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

        self._load_cn_data()

    def _load_cn_data(self):
        with open(config.TOPO_DICTIONARY_PATH, "r") as f:
            topo_data = json.load(f)
        with open(config.BB_DICTIONARY_PATH, "r") as f:
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
        self, matchmaker_result: Dict[str, Any]
    ) -> List[MOFComponent]:
        """Generate diverse combinations using stratified random sampling."""
        topo_list = matchmaker_result.get("topology", [])
        node_list = matchmaker_result.get("node", [])
        edge_list = matchmaker_result.get("edge", [])

        if not topo_list:
            topo_list = self._topo_list[:50]
        if not node_list:
            node_list = self._node_list[:100]
        if not edge_list:
            edge_list = self._edge_list[:50]

        return self._generate_diverse_combinations(
            topo_list, node_list, edge_list, max_combos=1000
        )

    def _generate_diverse_combinations(
        self,
        topo_list: List[str],
        node_list: List[str],
        edge_list: List[str],
        max_combos: int = 1000,
    ) -> List[MOFComponent]:
        """
        Generate combinations using stratified random sampling.

        Instead of sequential generation (which causes same topology/node in top results),
        we sample evenly across all 3 dimensions to ensure diversity.

        Algorithm:
        1. Calculate sample size per dimension: n^(1/3)
        2. Randomly sample from each dimension
        3. Generate all combinations from sampled subsets
        4. Shuffle final combinations
        """
        n_topo = len(topo_list)
        n_node = len(node_list)
        n_edge = len(edge_list)

        # Calculate samples per dimension (cube root for balanced coverage)
        # This ensures ~equal representation from each dimension
        samples_per_dim = max(1, int(round(max_combos ** (1 / 3))))

        n_sample_topo = min(samples_per_dim, n_topo)
        n_sample_node = min(samples_per_dim, n_node)
        n_sample_edge = min(samples_per_dim * 2, n_edge)  # Give more weight to edges

        # Stratified random sampling (without replacement)
        n_topo = len(topo_list)
        n_node = len(node_list)
        n_edge = len(edge_list)

        # Sample from each dimension
        sample_topo = (
            list(
                np.random.choice(
                    topo_list, size=min(n_sample_topo, n_topo), replace=False
                )
            )
            if n_topo > 0
            else []
        )

        sample_node = (
            list(
                np.random.choice(
                    node_list, size=min(n_sample_node, n_node), replace=False
                )
            )
            if n_node > 0
            else []
        )

        sample_edge = (
            list(
                np.random.choice(
                    edge_list, size=min(n_sample_edge, n_edge), replace=False
                )
            )
            if n_edge > 0
            else []
        )

        # Generate all combinations from sampled subsets WITH CN VALIDATION
        combinations = []
        for topo in sample_topo:
            topo_cn = self._topo_cn.get(topo, 0)
            for node in sample_node:
                node_cn = self._node_cn.get(node, 0)
                for edge in sample_edge:
                    edge_cn = self._edge_cn.get(edge, 0)

                    if topo_cn == node_cn and edge_cn == 2:
                        combinations.append(
                            MOFComponent(topology=topo, node=node, edge=edge)
                        )

        # If we need more combinations, add more random samples WITH CN VALIDATION
        if len(combinations) < max_combos:
            remaining = max_combos - len(combinations)
            extra_topo = list(
                np.random.choice(topo_list, size=min(remaining, n_topo), replace=False)
            )
            extra_node = list(
                np.random.choice(node_list, size=min(remaining, n_node), replace=False)
            )
            extra_edge = list(
                np.random.choice(edge_list, size=min(remaining, n_edge), replace=False)
            )

            for topo in extra_topo:
                topo_cn = self._topo_cn.get(topo, 0)
                for node in extra_node:
                    node_cn = self._node_cn.get(node, 0)
                    for edge in extra_edge:
                        edge_cn = self._edge_cn.get(edge, 0)
                        if len(combinations) >= max_combos:
                            break
                        if topo_cn == node_cn and edge_cn == 2:
                            combinations.append(
                                MOFComponent(topology=topo, node=node, edge=edge)
                            )
                    if len(combinations) >= max_combos:
                        break
                if len(combinations) >= max_combos:
                    break

        # Shuffle to ensure unbiased ranking
        np.random.shuffle(combinations)

        print(
            f"[Agent 3] Generated {len(combinations)} diverse combinations "
            f"(sampled from {len(topo_list)}×{len(node_list)}×{len(edge_list)} = "
            f"{len(topo_list) * len(node_list) * len(edge_list)} total)"
        )
        print(
            f"[Agent 3]   Sample coverage: {len(sample_topo)} topologies, "
            f"{len(sample_node)} nodes, {len(sample_edge)} edges"
        )

        return combinations[:max_combos]


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

        # Evaluate all combinations (no ranking, just scoring)
        all_mofs = []
        for comp, pred_geo in zip(combinations, predicted_geometries):
            score, match_details = self.ranker._calculate_match_score(
                pred_geo, target_geometry
            )
            # Check if ALL constraints are satisfied (score == 1.0)
            all_satisfied = all(
                v.startswith("✓")
                for k, v in match_details.items()
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
        with open(args.constraints, "r") as f:
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
        with open(args.matchmaker, "r") as f:
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
