"""
GA + NLP Surrogate for MOF H2 uptake optimization.

Crossover + Mutation from observed data:
  1. Crossover: pair observed MOFs, swap components -> check pool existence
  2. Mutation: 1-component change from observed MOFs -> check pool existence
  3. Union of candidates -> surrogate scoring -> top batch_size

Usage:
  conda run -n <env> python run_ga_mof2zeo.py --gpu 1
  conda run -n <env> python run_ga_mof2zeo.py --gpu 3 --n-runs 5
"""

import os
import sys
import json
import copy
import random
import gc
import argparse

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

PROJECT_ROOT = "<PROJECT_ROOT>"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core", "mof2zeo"))
from dataset import TOPO_DICT, NODE_DICT, EDGE_DICT, Scaler
from model import MOFEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POOL_PATH = os.path.join(SCRIPT_DIR, "candidate_pool.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ---------------------------------------------------------------------------
# Pool loading + index building
# ---------------------------------------------------------------------------

def load_pool(pool_path: str) -> dict:
    """Load pool and parse filenames. Returns {key: {topo, node, edge, target}}."""
    df = pd.read_csv(pool_path)
    pool = {}
    for _, row in df.iterrows():
        parts = row["filename"].split("+")
        if len(parts) != 3:
            continue
        topo, node, edge = parts
        if topo in TOPO_DICT and node in NODE_DICT and edge in EDGE_DICT:
            key = f"{topo}+{node}+{edge}"
            pool[key] = {"topo": topo, "node": node, "edge": edge, "target": row["target"]}
    return pool


def build_neighbor_index(pool_keys: set) -> tuple[dict, dict, dict]:
    """Build index for 1-component mutation neighbor lookup.

    Returns:
        topo_node_to_edges: {(topo, node): set of edges}
        topo_edge_to_nodes: {(topo, edge): set of nodes}
        node_edge_to_topos: {(node, edge): set of topos}
    """
    topo_node_to_edges = {}
    topo_edge_to_nodes = {}
    node_edge_to_topos = {}

    for key in pool_keys:
        t, n, e = key.split("+")
        topo_node_to_edges.setdefault((t, n), set()).add(e)
        topo_edge_to_nodes.setdefault((t, e), set()).add(n)
        node_edge_to_topos.setdefault((n, e), set()).add(t)

    return topo_node_to_edges, topo_edge_to_nodes, node_edge_to_topos


def find_crossover_children(observed_keys: list, evaluated_keys: set,
                            pool_keys: set) -> set:
    """Generate crossover children from all observed pairs and check pool existence.

    For each pair of observed MOFs, swap components to create children.
    Only returns unevaluated children that exist in pool.
    """
    children = set()
    observed = [k.split("+") for k in observed_keys]
    n = len(observed)

    for i in range(n):
        t1, n1, e1 = observed[i]
        for j in range(i + 1, n):
            t2, n2, e2 = observed[j]
            # All 2^3 = 8 combinations from two parents
            for t in (t1, t2):
                for nd in (n1, n2):
                    for e in (e1, e2):
                        k = f"{t}+{nd}+{e}"
                        if k not in evaluated_keys and k in pool_keys:
                            children.add(k)

    return children


def find_mutation_neighbors(observed_keys: list, evaluated_keys: set,
                            topo_node_to_edges: dict, topo_edge_to_nodes: dict,
                            node_edge_to_topos: dict) -> set:
    """Find all 1-component mutation neighbors of observed MOFs that exist in pool
    and haven't been evaluated yet."""
    neighbors = set()

    for key in observed_keys:
        t, n, e = key.split("+")

        # Vary edge (fix topo + node)
        for e2 in topo_node_to_edges.get((t, n), set()):
            k = f"{t}+{n}+{e2}"
            if k not in evaluated_keys:
                neighbors.add(k)

        # Vary node (fix topo + edge)
        for n2 in topo_edge_to_nodes.get((t, e), set()):
            k = f"{t}+{n2}+{e}"
            if k not in evaluated_keys:
                neighbors.add(k)

        # Vary topo (fix node + edge)
        for t2 in node_edge_to_topos.get((n, e), set()):
            k = f"{t2}+{n}+{e}"
            if k not in evaluated_keys:
                neighbors.add(k)

    return neighbors


# ---------------------------------------------------------------------------
# Surrogate (mof2zeo MOFEncoder, desc_dim=1)
# ---------------------------------------------------------------------------

def make_tensors(data_list: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert list of {topo, node, edge, target} to (x, y) tensors."""
    topo_ids = [TOPO_DICT[d["topo"]] for d in data_list]
    node_ids = [NODE_DICT[d["node"]] for d in data_list]
    edge_ids = [EDGE_DICT[d["edge"]] for d in data_list]
    targets = [d["target"] for d in data_list]
    x = torch.tensor(list(zip(topo_ids, node_ids, edge_ids)), dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    return x, y


def train_surrogate(observed_data: list, device: str = "cuda") -> tuple:
    """Train MOFEncoder(desc_dim=1) on observed data. Returns (model, scaler)."""
    targets = np.array([d["target"] for d in observed_data], dtype=np.float32)
    t_mean, t_std = float(targets.mean()), float(targets.std() + 1e-6)
    scaler = Scaler(np.array([t_mean]), np.array([t_std]), 0, 1)

    scaled_data = [{**d, "target": (d["target"] - t_mean) / t_std} for d in observed_data]

    n = len(scaled_data)
    n_train = max(int(n * 0.8), 2)
    idx = list(range(n))
    random.shuffle(idx)
    train_data = [scaled_data[i] for i in idx[:n_train]]
    valid_data = [scaled_data[i] for i in idx[n_train:]] if n_train < n else train_data[:2]

    x_train, y_train = make_tensors(train_data)
    x_valid, y_valid = make_tensors(valid_data)

    config = {
        "latent_dim": 128, "hid_dim1": 64, "hid_dim2": 32, "desc_dim": 1,
        "topo_size": len(TOPO_DICT), "node_size": len(NODE_DICT), "edge_size": len(EDGE_DICT),
    }
    model = MOFEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    batch_size = min(64, n_train)

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(x_train))
        for start in range(0, len(x_train), batch_size):
            bi = perm[start:start + batch_size]
            xb, yb = x_train[bi].to(device), y_train[bi].to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_valid.to(device)), y_valid.to(device)).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, scaler


def predict_surrogate(model: nn.Module, keys: list, scaler: Scaler,
                      device: str = "cuda") -> np.ndarray:
    """Predict H2 uptake for list of 'topo+node+edge' keys."""
    if not keys:
        return np.array([])
    records = []
    for k in keys:
        t, n, e = k.split("+")
        records.append([TOPO_DICT[t], NODE_DICT[n], EDGE_DICT[e]])
    x = torch.tensor(records, dtype=torch.long).to(device)
    with torch.no_grad():
        y_scaled = model(x).cpu().numpy().flatten()
    y = scaler.decode(y_scaled.reshape(1, -1)).flatten()
    return y


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(pool: dict, pool_keys: set, neighbor_idx: tuple,
            n_iter: int, batch_size: int, seed: int, device: str) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    topo_node_to_edges, topo_edge_to_nodes, node_edge_to_topos = neighbor_idx

    observed = []
    evaluated = set()
    all_targets = []
    history = {
        "iter": [], "selected_keys": [], "selected_targets": [],
        "cumulative_best": [], "cumulative_median": [],
        "iter_median": [], "iter_max": [], "iter_mean": [], "n_evaluated": [],
        "n_crossover": [], "n_mutation": [], "n_candidates": [],
    }

    for it in range(n_iter):
        if it == 0:
            # Random initialization
            available = list(pool_keys - evaluated)
            selected_keys = random.sample(available, min(batch_size, len(available)))
            history["n_crossover"].append(0)
            history["n_mutation"].append(0)
            history["n_candidates"].append(0)
        else:
            # Train surrogate on observed data
            print(f"    Training surrogate ({len(observed)} samples)...")
            model, scaler = train_surrogate(observed, device)

            observed_keys = [f"{d['topo']}+{d['node']}+{d['edge']}" for d in observed]

            # Step 1: Crossover — pair observed MOFs, swap components
            crossover_children = find_crossover_children(observed_keys, evaluated, pool_keys)
            n_crossover = len(crossover_children)

            # Step 2: Mutation — 1-component change from observed MOFs
            mutation_neighbors = find_mutation_neighbors(
                observed_keys, evaluated,
                topo_node_to_edges, topo_edge_to_nodes, node_edge_to_topos,
            )
            n_mutation = len(mutation_neighbors)

            # Step 3: Union (crossover ⊂ mutation in practice, but structurally correct)
            candidates = crossover_children | mutation_neighbors
            n_candidates = len(candidates)
            print(f"    Crossover: {n_crossover}, Mutation: {n_mutation}, "
                  f"Union: {n_candidates}")

            # Step 4: Score all candidates with surrogate -> top batch_size
            if candidates:
                cand_list = list(candidates)
                scores = predict_surrogate(model, cand_list, scaler, device)
                order = np.argsort(scores)[::-1]
                selected_keys = [cand_list[order[i]] for i in range(min(batch_size, len(order)))]
            else:
                selected_keys = []

            # Pad with random if not enough candidates
            if len(selected_keys) < batch_size:
                avail = list(pool_keys - evaluated - set(selected_keys))
                need = batch_size - len(selected_keys)
                if avail:
                    selected_keys.extend(random.sample(avail, min(need, len(avail))))
                print(f"    Padded {need} random (had {n_candidates} candidates)")

            history["n_crossover"].append(n_crossover)
            history["n_mutation"].append(n_mutation)
            history["n_candidates"].append(n_candidates)

            del model
            cleanup_gpu()

        # Evaluate selected keys
        targets = [pool[k]["target"] for k in selected_keys]
        for k in selected_keys:
            evaluated.add(k)
            observed.append(pool[k])
        all_targets.extend(targets)

        cum_best = max(all_targets)
        cum_med = float(np.median(all_targets))
        it_med = float(np.median(targets))
        it_max = float(np.max(targets))
        it_mean = float(np.mean(targets))

        history["iter"].append(it)
        history["selected_keys"].append(selected_keys)
        history["selected_targets"].append(targets)
        history["cumulative_best"].append(cum_best)
        history["cumulative_median"].append(cum_med)
        history["iter_median"].append(it_med)
        history["iter_max"].append(it_max)
        history["iter_mean"].append(it_mean)
        history["n_evaluated"].append(len(evaluated))

        print(f"    Iter {it}: median={it_med:.2f}, max={it_max:.2f}, "
              f"cum_best={cum_best:.2f}, cum_median={cum_med:.2f}, n_eval={len(evaluated)}")

    history["seed"] = seed
    return history


def main():
    parser = argparse.ArgumentParser(description="GA + mof2zeo surrogate (fair 1-component mutation)")
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GA+mof2zeo] Device: {device} (GPU {args.gpu})")

    pool = load_pool(POOL_PATH)
    pool_keys = set(pool.keys())
    topos = set(v["topo"] for v in pool.values())
    nodes = set(v["node"] for v in pool.values())
    edges = set(v["edge"] for v in pool.values())
    print(f"[GA+mof2zeo] Pool: {len(pool)} entries")
    print(f"[GA+mof2zeo] Vocab: {len(topos)} topos, {len(nodes)} nodes, {len(edges)} edges")
    print(f"[GA+mof2zeo] Target: mean={np.mean([v['target'] for v in pool.values()]):.2f}, "
          f"max={np.max([v['target'] for v in pool.values()]):.2f}")

    # Build neighbor index once
    print("[GA+mof2zeo] Building neighbor index...")
    neighbor_idx = build_neighbor_index(pool_keys)
    tn2e, te2n, ne2t = neighbor_idx
    print(f"[GA+mof2zeo] Index: {len(tn2e)} (t,n)->e, {len(te2n)} (t,e)->n, {len(ne2t)} (n,e)->t")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_runs = []
    for run_i in range(args.n_runs):
        seed = args.seed + run_i * 100
        print(f"\n[GA+mof2zeo] === Run {run_i + 1}/{args.n_runs} (seed={seed}) ===")
        history = run_one(pool, pool_keys, neighbor_idx,
                          args.n_iter, args.batch_size, seed, device)
        all_runs.append(history)
        cleanup_gpu()

    # Save
    results_path = os.path.join(RESULTS_DIR, f"ga_mof2zeo_b{args.batch_size}_n{args.n_iter}.json")
    with open(results_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\n[GA+mof2zeo] Results saved: {results_path}")

    cum_bests = [r["cumulative_best"][-1] for r in all_runs]
    cum_meds = [r["cumulative_median"][-1] for r in all_runs]
    print(f"\n[GA+mof2zeo] === Summary ({args.n_runs} runs) ===")
    print(f"  Final cum_best:   {np.mean(cum_bests):.2f} +/- {np.std(cum_bests):.2f}")
    print(f"  Final cum_median: {np.mean(cum_meds):.2f} +/- {np.std(cum_meds):.2f}")

    # Print candidate stats
    for run in all_runs:
        xo = run["n_crossover"]
        mut = run["n_mutation"]
        cand = run["n_candidates"]
        print(f"  Seed {run['seed']}: crossover={xo}, mutation={mut}, union={cand}")


if __name__ == "__main__":
    main()
