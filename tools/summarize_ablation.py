#!/usr/bin/env python3
"""Summarize ablation results into a compact CSV table.

Example:
    python tools/summarize_ablation.py \
        --ablation_dir ./checkpoints/ablation \
        --output ablation_results.csv
"""

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


def _append_result(results_data, method, rank1="-", rank5="-", map_value="-", status="done"):
    results_data.append(
        {
            "Method": method,
            "Rank-1": _format_metric(rank1),
            "Rank-5": _format_metric(rank5),
            "mAP": _format_metric(map_value),
            "Status": status,
        }
    )


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

    print("\n" + "=" * 60)
    print("  Ablation Results Summary")
    print("=" * 60 + "\n")

    for exp_name, exp_dir_name in experiments:
        if exp_dir_name == "atrw_full":
            v2_dir = Path("./checkpoints/atrw_v2")
            if v2_dir.exists():
                print(f"INFO: experiment 5 uses the V2 result: {v2_dir}")
                results = load_results(v2_dir)
                if results:
                    rank1 = results.get("rank1", results.get("Rank-1", "-"))
                    rank5 = results.get("rank5", results.get("Rank-5", "-"))
                    map_value = results.get("mAP", results.get("map", "-"))
                    _append_result(results_data, exp_name, rank1, rank5, map_value, "done (using V2 result)")
                    print(f"[OK] {exp_name:20s} | Rank-1: {_format_metric(rank1):>8s} | mAP: {_format_metric(map_value):>8s}")
                    continue

        exp_dir = ablation_dir / exp_dir_name
        if not exp_dir.exists():
            print(f"WARNING: experiment directory does not exist: {exp_dir}")
            _append_result(results_data, exp_name, status="not run")
            continue

        results = load_results(exp_dir)
        if results is None:
            _append_result(results_data, exp_name, status="missing result")
            continue

        rank1 = results.get("rank1", results.get("Rank-1", "-"))
        rank5 = results.get("rank5", results.get("Rank-5", "-"))
        map_value = results.get("mAP", results.get("map", "-"))

        _append_result(results_data, exp_name, rank1, rank5, map_value)
        print(f"[OK] {exp_name:20s} | Rank-1: {_format_metric(rank1):>8s} | mAP: {_format_metric(map_value):>8s}")

    df = pd.DataFrame(results_data)

    if len(df) > 1:
        print("\n" + "-" * 60)
        print("  Incremental Analysis")
        print("-" * 60 + "\n")

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

    print("\n" + "=" * 60)
    print("  Full Results Table")
    print("=" * 60 + "\n")
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Summary complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
