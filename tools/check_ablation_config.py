#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import sys
from pathlib import Path
from typing import Dict, Any

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_nested_value(config: dict, keys: list):
    value = config
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return None


def main():
    config_dir = Path('config/ablation')

    experiments = [
        ('Exp1_Baseline', 'atrw_ablation_1_baseline.yaml'),
        ('Exp2_IPAID', 'atrw_ablation_2_ipaid.yaml'),
        ('Exp3_FGID', 'atrw_ablation_3_fgid.yaml'),
        ('Exp4_IICL', 'atrw_ablation_4_iicl.yaml'),
        ('Exp5_Full', 'atrw_ablation_5_full.yaml'),
    ]

    print("\n" + "="*100)
    print("  Ablation Study Configuration Validation")
    print("="*100 + "\n")

    configs = {}
    for name, filename in experiments:
        config_path = config_dir / filename
        if config_path.exists():
            configs[name] = load_config(config_path)
        else:
            print(f"[ERROR] Config not found: {config_path}")

    print("Module Configuration Comparison:\n")
    print(f"{'Module':<40} | {'Exp1':<10} | {'Exp2':<10} | {'Exp3':<10} | {'Exp4':<10} | {'Exp5':<10}")
    print("-" * 100)

    checks = [
        ('Illumination Module Enabled', ['model', 'illumination_module', 'enabled']),
        ('IPAID: use_sensitivity', ['illumination_module', 'module_params', 'use_sensitivity']),
        ('IPAID: use_refinement', ['illumination_module', 'module_params', 'use_refinement']),
        ('FGID: use_feature_guided', ['illumination_module', 'module_params', 'use_feature_guided']),
        ('Color Illumination', ['illumination_module', 'module_params', 'use_color_illumination']),
        ('IICL Enabled', ['training', 'iicl', 'enabled']),
        ('Recon Loss Weight', ['illumination_module', 'loss_params', 'lambda_recon']),
    ]

    for check_name, keys in checks:
        row = f"{check_name:<40} |"
        for exp_name, _ in experiments:
            if exp_name not in configs:
                row += f" {'N/A':<10} |"
                continue

            value = get_nested_value(configs[exp_name], keys)

            if value is None:
                display = "N/A"
            elif isinstance(value, bool):
                display = "YES" if value else "NO"
            elif isinstance(value, (int, float)):
                display = f"{value:.2f}" if value > 0 else "0.00"
            else:
                display = str(value)

            row += f" {display:<10} |"

        print(row)

    print("\n" + "="*100)
    print("  Strictness Validation")
    print("="*100 + "\n")

    validations = [
        ('Exp1 -> Exp2: Add IPAID only', 'Exp1_Baseline', 'Exp2_IPAID', [
            (['model', 'illumination_module', 'enabled'], False, True),
            (['illumination_module', 'module_params', 'use_feature_guided'], False, False),
            (['training', 'iicl', 'enabled'], False, False),
        ]),
        ('Exp2 -> Exp3: Add FGID only', 'Exp2_IPAID', 'Exp3_FGID', [
            (['illumination_module', 'module_params', 'use_feature_guided'], False, True),
            (['training', 'iicl', 'enabled'], False, False),
        ]),
        ('Exp3 -> Exp4: Add IICL only', 'Exp3_FGID', 'Exp4_IICL', [
            (['training', 'iicl', 'enabled'], False, True),
            (['illumination_module', 'module_params', 'use_feature_guided'], True, True),
        ]),
        ('Exp4 -> Exp5: Should be identical', 'Exp4_IICL', 'Exp5_Full', [
            (['training', 'iicl', 'enabled'], True, True),
            (['illumination_module', 'module_params', 'use_feature_guided'], True, True),
        ]),
    ]

    all_valid = True

    for desc, exp1_name, exp2_name, checks in validations:
        print(f"[CHECK] {desc}")

        if exp1_name not in configs or exp2_name not in configs:
            print(f"  [WARN] Config missing, skip\n")
            continue

        config1 = configs[exp1_name]
        config2 = configs[exp2_name]

        valid = True
        for keys, expected1, expected2 in checks:
            val1 = get_nested_value(config1, keys)
            val2 = get_nested_value(config2, keys)

            key_str = '.'.join(keys)

            if val1 == expected1 and val2 == expected2:
                print(f"  [OK] {key_str}: {val1} -> {val2}")
            else:
                print(f"  [FAIL] {key_str}: Expected {expected1} -> {expected2}, Got {val1} -> {val2}")
                valid = False
                all_valid = False

        if valid:
            print(f"  [PASS]\n")
        else:
            print(f"  [FAIL]\n")

    print("="*100)
    if all_valid:
        print("[SUCCESS] All ablation configurations are strict and correct!")
    else:
        print("[ERROR] Some configurations are not strict, please check!")
    print("="*100 + "\n")

    print("Ablation Study Design:")
    print("  Exp1: Baseline (OSNet-AIN only)")
    print("  Exp2: + IPAID (Add illumination decomposition)")
    print("  Exp3: + FGID (Add feature-guided refinement) <- CORE INNOVATION")
    print("  Exp4: + IICL (Add illumination-variant feature consistency)")
    print("  Exp5: Full Model (Same as Exp4, use pre-trained model)")
    print()


if __name__ == '__main__':
    main()
