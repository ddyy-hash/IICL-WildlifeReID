#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ablation study results")
    parser.add_argument(
        "--ablation_dir",
        type=str,
        default="./checkpoints/ablation",
        help="Directory containing ablation experiments",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ablation_results.csv",
        help="Output CSV file",
    )
    return parser.parse_args()


def load_results(exp_dir):
    results_file = Path(exp_dir) / "results.json"

    if not results_file.exists():
        print(f"WARNING: results file does not exist: {results_file}")
        return None

    with results_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_metric(value):
    return f"{value:.2f}%" if isinstance(value, (int, float)) else value


def _extract_retrieval_metrics(results):
    if "openset" in results and "single_camera" in results["openset"]:
        metrics = results["openset"]["single_camera"]
    elif "closedset" in results:
        metrics = results["closedset"]
    else:
        metrics = results

    rank1 = metrics.get("rank1", metrics.get("Rank-1", "-"))
    rank5 = metrics.get("rank5", metrics.get("Rank-5", "-"))
    map_value = metrics.get("mAP", metrics.get("map", "-"))
    return rank1, rank5, map_value


def main():
    args = parse_args()
    ablation_dir = Path(args.ablation_dir)

    if not ablation_dir.exists():
        print(f"ERROR: ablation directory does not exist: {ablation_dir}")
        return

    experiments = [
        ("Baseline", "atrw_baseline"),
        ("+ IPAID", "atrw_ipaid"),
        ("+ FGID", "atrw_fgid"),
        ("+ IICL", "atrw_iicl"),
        ("Full Model", "atrw_full"),
    ]

    results_data = []

    print("\n" + "=" * 80)
    print("  Ablation Results Summary")
    print("=" * 80 + "\n")

    for exp_name, exp_dir_name in experiments:
        if exp_dir_name == "atrw_full":
            exp_dir = ablation_dir / exp_dir_name
            if not exp_dir.exists():
                v2_dir = Path("./checkpoints/atrw_v2")
                if v2_dir.exists():
                    print(f"INFO: experiment 5 uses the V2 result: {v2_dir}")
                    exp_dir = v2_dir
        else:
            exp_dir = ablation_dir / exp_dir_name

        if not exp_dir.exists():
            print(f"WARNING: experiment directory does not exist: {exp_dir}")
            results_data.append(
                {"Method": exp_name, "Rank-1": "-", "Rank-5": "-", "mAP": "-", "Status": "not run"}
            )
            continue

        results = load_results(exp_dir)
        if results is None:
            print(f"WARNING: {exp_name:20s} | evaluation result not found")
            results_data.append(
                {"Method": exp_name, "Rank-1": "-", "Rank-5": "-", "mAP": "-", "Status": "not evaluated"}
            )
            continue

        rank1, rank5, map_value = _extract_retrieval_metrics(results)
        results_data.append(
            {
                "Method": exp_name,
                "Rank-1": _format_metric(rank1),
                "Rank-5": _format_metric(rank5),
                "mAP": _format_metric(map_value),
                "Status": "done",
            }
        )

        if isinstance(rank1, (int, float)) and isinstance(map_value, (int, float)):
            print(f"[OK] {exp_name:20s} | Rank-1: {rank1:6.2f}% | mAP: {map_value:6.2f}%")
        else:
            print(f"WARNING: {exp_name:20s} | unexpected data format")

    df = pd.DataFrame(results_data)

    if len(df) > 1:
        print("\n" + "-" * 80)
        print("  Incremental Analysis")
        print("-" * 80 + "\n")

        for i in range(1, len(df)):
            if df.loc[i, "Status"] != "done" or df.loc[i - 1, "Status"] != "done":
                continue

            try:
                current_map = float(df.loc[i, "mAP"].rstrip("%"))
                previous_map = float(df.loc[i - 1, "mAP"].rstrip("%"))
            except ValueError:
                continue

            delta = current_map - previous_map
            print(f"{df.loc[i, 'Method']:20s} vs {df.loc[i - 1, 'Method']:20s}: delta mAP = {delta:+.2f}%")

            if "+ FGID" in df.loc[i, "Method"]:
                print(f"  FGID contribution: {delta:+.2f}%")
                if delta >= 2.0:
                    print("  Interpretation: strong incremental gain")
                elif delta >= 1.5:
                    print("  Interpretation: moderate incremental gain")
                else:
                    print("  Interpretation: small incremental gain")

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 80)
    print("  Full Results Table")
    print("=" * 80 + "\n")
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("  Summary complete.")
    print("=" * 80 + "\n")

    print("Notes:")
    print("  - Results come from eval_atrw_closedset.py and eval_atrw_openset.py")
    print("  - If a row shows 'not evaluated', run the matching evaluation command first")
    print("  - The primary metric of interest is the open-set single-camera mAP")
    print()


if __name__ == "__main__":
    main()
