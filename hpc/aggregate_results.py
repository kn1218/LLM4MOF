#!/usr/bin/env python3
"""
Aggregate per-MOF result JSONs into batch_results.json.

Usage:
  python aggregate_results.py --manifest batch_manifest.json --results-dir results/
"""

import argparse
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description="Aggregate MOF simulation results")
    parser.add_argument("--manifest", required=True, help="Path to batch_manifest.json")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    n_success = 0
    n_fail = 0
    n_missing = 0

    for job in manifest["jobs"]:
        filename = job["filename"]
        result_file = os.path.join(args.results_dir, f"{filename}.json")

        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            results.append(result)
            if result.get("status") == "success":
                n_success += 1
            else:
                n_fail += 1
        else:
            # Job didn't produce output — mark as failed
            results.append({
                "filename": filename,
                "topology": job["topology"],
                "node": job["node"],
                "edge": job["edge"],
                "beam_id": job["beam_id"],
                "predicted_geometry": job.get("predicted_geometry"),
                "match_score": job.get("match_score", 0.0),
                "status": "missing",
                "real_uptake": None,
                "error_msg": "No result file produced by HPC job",
                "wall_seconds": 0.0,
                "stages": {},
            })
            n_missing += 1

    batch_results = {
        "experiment_id": manifest["experiment_id"],
        "iteration": manifest["iteration"],
        "n_jobs": manifest["n_jobs"],
        "n_success": n_success,
        "n_fail": n_fail,
        "n_missing": n_missing,
        "total_wall_seconds": sum(r.get("wall_seconds", 0) for r in results),
        "aggregated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }

    output_path = os.path.join(args.results_dir, "batch_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)

    print(f"[Aggregate] {n_success} success, {n_fail} fail, {n_missing} missing")
    print(f"[Aggregate] Written to {output_path}")


if __name__ == "__main__":
    main()
