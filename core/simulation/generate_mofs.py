#!/usr/bin/env python3
"""
Generate MOF CIF files from test_result_agent3.json using PORMAKE

Usage:
    python generate_mofs.py [--mof-dir DIR] [--max MOFs]

Requirements:
    - PORMAKE installed (pip install git+https://github.com/Sangwon91/PORMAKE.git)
    - Database files in default location or --db-path specified
"""

import os
import sys
import json
import argparse

RESULT_FILE = None
DEFAULT_OUTPUT_DIR = None


def parse_mof_filename(filename):
    """Parse topology+node+edge from filename

    Example: qzd+N32+E17 -> topology="qzd", node="N32", edge="E17"
    """
    parts = filename.split("+")
    if len(parts) != 3:
        raise ValueError(f"Invalid filename format: {filename}")

    topology, node, edge = parts

    if not node.startswith("N"):
        raise ValueError(f"Invalid node format: {node}")
    if not edge.startswith("E"):
        raise ValueError(f"Invalid edge format: {edge}")

    return {
        "topology": topology,
        "node": node,
        "edge": edge,
    }


def load_mof_data():
    """Load top 20 MOFs from test_result_agent3.json"""
    with open(RESULT_FILE, "r") as f:
        data = json.load(f)
    return data["proposals"]["ranked_mofs"][:20]


def generate_mof(mof_info, database, builder, output_dir):
    """Generate CIF file for a single MOF using PORMAKE"""
    filename = mof_info["filename"]
    topology_name = mof_info["topology"]
    node_name = mof_info["node"]
    edge_name = mof_info["edge"]

    output_file = os.path.join(output_dir, f"{filename}.cif")

    if os.path.exists(output_file):
        print(f"   [SKIP] {filename}.cif already exists")
        return True

    try:
        print(f"   Loading: topo={topology_name}, node={node_name}, edge={edge_name}")

        topology = database.get_topo(topology_name)
        node_bb = database.get_bb(node_name)
        edge_bb = database.get_bb(edge_name)

        topology.describe()

        # Safety check: multi-node-type topologies crash build_by_type
        # because we only assign node_bbs = {0: node_bb}.
        # HAN_SAFE_TOPOS should filter these out upstream, but guard here too.
        # NOTE: topology.cn includes BOTH node and edge connectivities
        # (edges always have cn=2), so we use topology.n_node_types instead.
        if topology.n_node_types > 1:
            print(f"   [SKIP] {filename}: multi-node-type topology "
                  f"({topology.n_node_types} node types)")
            return False

        node_bbs = {0: node_bb}

        cn = topology.cn[0]

        edge_bbs = {(0, 0): edge_bb}

        mof = builder.build_by_type(
            topology=topology, node_bbs=node_bbs, edge_bbs=edge_bbs
        )

        mof.write_cif(output_file)
        print(f"   [OK] Saved: {filename}.cif")
        return True

    except KeyError as e:
        print(f"   [SKIP] {filename}: KeyError during build (likely multi-node-type): {e}")
        return False
    except Exception as e:
        print(f"   [ERROR] {filename}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate MOF CIF files using PORMAKE")
    parser.add_argument(
        "--mof-dir", required=True, help="Output directory for CIF files"
    )
    parser.add_argument(
        "--result_file", required=True, help="Input JSON file with ranked MOFs"
    )
    parser.add_argument(
        "--max", type=int, default=20, help="Maximum number of MOFs to generate"
    )
    parser.add_argument(
        "--db-path", default=None, help="Path to PORMAKE database (optional)"
    )

    args = parser.parse_args()

    global RESULT_FILE, DEFAULT_OUTPUT_DIR
    RESULT_FILE = args.result_file
    DEFAULT_OUTPUT_DIR = args.mof_dir

    print("=" * 70)
    print("MOF Generator using PORMAKE")
    print("=" * 70)

    print(f"\n[1] Loading PORMAKE...")
    import warnings

    warnings.filterwarnings("ignore")

    import importlib

    pm = importlib.import_module("pormake")

    print(f"    Database: (default)")

    database = pm.Database()
    builder = pm.Builder()

    print(f"\n[2] Loading MOF data from: {RESULT_FILE}")
    mofs = load_mof_data()[: args.max]
    print(f"    Loaded {len(mofs)} MOFs")

    os.makedirs(args.mof_dir, exist_ok=True)

    print(f"\n[3] Generating CIF files...")
    print(f"    Output directory: {args.mof_dir}")

    success = 0
    failed = 0

    for i, mof in enumerate(mofs):
        print(f"\n    [{i + 1}/{len(mofs)}] {mof['filename']}")

        try:
            mof_info = parse_mof_filename(mof["filename"])
            mof_info["rank"] = mof["rank"]
            mof_info["filename"] = mof["filename"]

            if generate_mof(mof_info, database, builder, args.mof_dir):
                success += 1
            else:
                failed += 1

        except Exception as e:
            print(f"   [ERROR] {e}")
            failed += 1

    print(f"\n[4] Summary:")
    print(f"    Success: {success}")
    print(f"    Failed: {failed}")
    print(f"    Output: {args.mof_dir}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
