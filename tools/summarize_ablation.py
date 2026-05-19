#!/usr/bin/env python3
"""

    python tools/summarize_ablation.py \
        --ablation_dir ./checkpoints/ablation \
        --output ablation_results.csv
"""

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

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


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

    print("\n" + "="*60)
    print("  Ablation results summary")
    print("="*60 + "\n")

    for exp_name, exp_dir_name in experiments:
        if exp_dir_name == 'atrw_full':
            v2_dir = Path('./checkpoints/atrw_v2')
            if v2_dir.exists():
                print(f"INFO: experiment 5 uses the V2 result: {v2_dir}")
                results = load_results(v2_dir)
                if results:
                    rank1 = results.get('rank1', results.get('Rank-1', '-'))
                    rank5 = results.get('rank5', results.get('Rank-5', '-'))
                    mAP = results.get('mAP', results.get('map', '-'))
                    results_data.append({
                        'Method': exp_name,
                        'Rank-1': f"{rank1:.2f}%" if isinstance(rank1, (int, float)) else rank1,
                        'Rank-5': f"{rank5:.2f}%" if isinstance(rank5, (int, float)) else rank5,
                        'mAP': f"{mAP:.2f}%" if isinstance(mAP, (int, float)) else mAP,
                        'Status': 'done (using V2 result)'
                    })
                    print(f"{exp_name:20s} | Rank-1: {rank1:6.2f}% | mAP: {mAP:6.2f}% (from V2)")
                    continue

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
                'Status': 'missing result'
            })
            continue

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

        print(f"✓ {exp_name:20s} | Rank-1: {rank1 if isinstance(rank1, str) else f'{rank1:.2f}%':>8s} | mAP: {mAP if isinstance(mAP, str) else f'{mAP:.2f}%':>8s}")

    df = pd.DataFrame(results_data)

    if len(df) > 1:
        print("\n" + "-"*60)
        print("  Incremental analysis")
        print("-"*60 + "\n")

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

    print("\n" + "="*60)
    print("  Full results table")
    print("="*60 + "\n")
    print(df.to_string(index=False))

    print("\n" + "="*60)
    print("  Summary complete.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
