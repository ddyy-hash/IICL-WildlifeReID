#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize ablation study results')
    parser.add_argument('--ablation_dir', type=str, default='./checkpoints/ablation',
                        help='Directory containing ablation experiments')
    parser.add_argument('--output', type=str, default='ablation_results.csv',
                        help='Output CSV file')
    return parser.parse_args()


def load_results(exp_dir):
    results_file = Path(exp_dir) / 'results.json'

    if not results_file.exists():
        print(f"WARNING: results file does not exist: {results_file}")
        return None

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results


def extract_metrics_from_submission(submission_file):
    if not Path(submission_file).exists():
        return None

    return None


def main():
    args = parse_args()

    ablation_dir = Path(args.ablation_dir)

    if not ablation_dir.exists():
        print(f"ERROR: ablation directory does not exist: {ablation_dir}")
        return

    experiments = [
        ('Baseline', 'atrw_baseline'),
        ('+ IPAID', 'atrw_ipaid'),
        ('+ FGID', 'atrw_fgid'),
        ('+ IICL', 'atrw_iicl'),
        ('Full Model', 'atrw_full'),
    ]

    results_data = []

    print("\n" + "="*80)
    print("  Ablation results summary (based on real evaluation)")
    print("="*80 + "\n")

    for exp_name, exp_dir_name in experiments:
        if exp_dir_name == 'atrw_full':
            exp_dir = ablation_dir / exp_dir_name
            if not exp_dir.exists():
                v2_dir = Path('./checkpoints/atrw_v2')
                if v2_dir.exists():
                    print(f"INFO: experiment 5 uses the V2 result: {v2_dir}")
                    exp_dir = v2_dir

        else:
            exp_dir = ablation_dir / exp_dir_name

        if not exp_dir.exists():
            print(f"WARNING: experiment directory does not exist: {exp_dir}")
            results_data.append({
                'Method': exp_name,
                'Rank-1': '-',
                'Rank-5': '-',
                'mAP': '-',
                'Status': 'not run'
            })
            continue

        results = load_results(exp_dir)

        if results is None:
            results_data.append({
                'Method': exp_name,
                'Rank-1': '-',
                'Rank-5': '-',
                'mAP': '-',
                'Status': 'not evaluated'
            })
            print(f"WARNING: {exp_name:20s} | evaluation result not found")
            continue

        # {
        #   "closedset": {"rank1": 98.15, "rank5": 99.57, "mAP": 81.38},
        #   "openset": {"single_camera": {"rank1": 88.45, "mAP": 71.73}, ...}
        # }

        if 'openset' in results and 'single_camera' in results['openset']:
            metrics = results['openset']['single_camera']
            rank1 = metrics.get('rank1', metrics.get('Rank-1', '-'))
            rank5 = metrics.get('rank5', metrics.get('Rank-5', '-'))
            mAP = metrics.get('mAP', metrics.get('map', '-'))
        elif 'closedset' in results:
            metrics = results['closedset']
            rank1 = metrics.get('rank1', metrics.get('Rank-1', '-'))
            rank5 = metrics.get('rank5', metrics.get('Rank-5', '-'))
            mAP = metrics.get('mAP', metrics.get('map', '-'))
        else:
            rank1 = results.get('rank1', results.get('Rank-1', '-'))
            rank5 = results.get('rank5', results.get('Rank-5', '-'))
            mAP = results.get('mAP', results.get('map', '-'))

        results_data.append({
            'Method': exp_name,
            'Rank-1': f"{rank1:.2f}%" if isinstance(rank1, (int, float)) else rank1,
            'Rank-5': f"{rank5:.2f}%" if isinstance(rank5, (int, float)) else rank5,
            'mAP': f"{mAP:.2f}%" if isinstance(mAP, (int, float)) else mAP,
            'Status': 'done'
        })

        if isinstance(rank1, (int, float)) and isinstance(mAP, (int, float)):
            print(f"✓ {exp_name:20s} | Rank-1: {rank1:6.2f}% | mAP: {mAP:6.2f}%")
        else:
            print(f"WARNING: {exp_name:20s} | unexpected data format")

    df = pd.DataFrame(results_data)

    if len(df) > 1:
        print("\n" + "-"*80)
        print("  Incremental analysis")
        print("-"*80 + "\n")

        for i in range(1, len(df)):
            if df.loc[i, 'Status'] == 'done' and df.loc[i-1, 'Status'] == 'done':
                try:
                    curr_mAP = float(df.loc[i, 'mAP'].rstrip('%'))
                    prev_mAP = float(df.loc[i-1, 'mAP'].rstrip('%'))
                    delta = curr_mAP - prev_mAP

                    print(f"{df.loc[i, 'Method']:20s} vs {df.loc[i-1, 'Method']:20s}: Δ mAP = {delta:+.2f}%")

                    if '+ FGID' in df.loc[i, 'Method']:
                        print(f"  FGID contribution: {delta:+.2f}%")
                        if delta >= 2.0:
                            print("  Recommendation: submit to ACM MM (estimated acceptance 40-45%)")
                        elif delta >= 1.5:
                            print("  Recommendation: submission is possible, but risk is higher (estimated acceptance 30-35%)")
                        else:
                            print("  Recommendation: not advised for ACM MM (estimated acceptance <25%)")
                except:
                    pass

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    print("\n" + "="*80)
    print("  Full results table")
    print("="*80 + "\n")
    print(df.to_string(index=False))

    print("\n" + "="*80)
    print("  Summary complete.")
    print("="*80 + "\n")

    print("Notes:")
    print("  - Results come from eval_atrw_closedset.py and eval_atrw_openset.py")
    print("  - If a row shows 'not evaluated', run: bash evaluate_ablation.sh <exp_name> <checkpoint>")
    print("  - The primary metric of interest is the open-set single-camera mAP")
    print()


if __name__ == '__main__':
    main()
