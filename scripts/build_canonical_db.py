"""Build the CANONICAL PORMAKE markscheme DBs by filtering to the shared building-block whitelist.

Single source of truth = core/mof2zeo/data/{node,edge,topology}.txt (the retrained mof2zeo vocabulary).
A MOF is kept iff it is single-node single-edge (topo+N+E) AND topo/node/edge are all in the whitelist.
This physically replaces the 6 PORMAKE H2 CSVs in data/ (originals backed up to data/_full_backup/,
gitignored). Re-run any time the whitelist changes.

Usage:  PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 python scripts/build_canonical_db.py
"""
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
BBLIST = os.path.join(ROOT, "core", "mof2zeo", "data")  # single source = retrained mof2zeo vocab
BACKUP = os.path.join(DATA, "_full_backup")

DBS = [
    "total_characteristics_h2_100bar_77K.csv",
    "total_characteristics_h2_100bar_77K_gperL.csv",
    "total_characteristics_h2_100bar_77K_mol_kg.csv",
    "total_characteristics_h2_5bar_77K.csv",
    "total_characteristics_h2_5bar_77K_gperL.csv",
    "total_characteristics_h2_5bar_77K_mol_kg.csv",
]


def _load(name):
    with open(os.path.join(BBLIST, name), encoding="utf-8") as f:
        return frozenset(l.strip() for l in f if l.strip())


def main():
    nodes, edges, topos = _load("node.txt"), _load("edge.txt"), _load("topology.txt")
    print(f"whitelist: {len(topos)} topologies, {len(nodes)} nodes, {len(edges)} edges")
    os.makedirs(BACKUP, exist_ok=True)

    def keep(fn):
        p = str(fn).split("+")
        return len(p) == 3 and p[0] in topos and p[1] in nodes and p[2] in edges

    for name in DBS:
        src = os.path.join(DATA, name)
        if not os.path.exists(src):
            print(f"[skip] missing {name}"); continue
        # back up the pristine original once, then always filter FROM the backup (idempotent, regenerable)
        bak = os.path.join(BACKUP, name)
        if not os.path.exists(bak):
            shutil.copy2(src, bak)
        d = pd.read_csv(bak, encoding="utf-8")
        kept = d[d["filename"].apply(keep)].copy()
        kept.to_csv(src, index=False, encoding="utf-8")
        print(f"[{name}] {len(d):,} -> {len(kept):,} kept ({len(d) - len(kept):,} removed)")
    print("\nCanonical DBs written (originals preserved in data/_full_backup/).")


if __name__ == "__main__":
    main()
