#!/usr/bin/env python3
"""
ATRW 双协议评估主脚本

同时运行两种评估协议，生成可发表的完整结果：

1. Closed-Set (7:3 Split) - 与现有论文直接对比
   - 同一 107 个 ID 按 7:3 划分
   - 与 SMFFEN 2024 等论文使用相同协议
   
2. Open-Set (官方协议) - 展示真实泛化能力
   - 训练 107 个 ID，测试 75 个不同的 ID
   - 使用官方 ATRWEvalScript 评估

使用方法:
    python tools/eval_atrw_both.py --checkpoint checkpoints/your_model.pth

输出格式 (适合论文表格):
    
    Table: Comparison on ATRW Dataset
    ============================================================
    Protocol      | Method          | Rank-1  | Rank-5  | mAP
    ============================================================
    Closed-Set    | ResNet50        | 91.70%  | 97.90%  | 68.40%
    (7:3 Split)   | PCB             | 94.70%  | 98.40%  | 71.20%
                  | SMFFEN 2024     | 96.30%  | 98.90%  | 78.70%
                  | Ours (IICL)     | XX.XX%  | XX.XX%  | XX.XX%
    ------------------------------------------------------------
    Open-Set      | Ours (IICL)     | XX.XX%  | XX.XX%  | XX.XX%
    (Official)    | (single-cam)    |         |         |
                  | (cross-cam)     | XX.XX%  | XX.XX%  | XX.XX%
    ============================================================
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='ATRW Dual Protocol Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--backbone', type=str, default='osnet_ain_x1_0',
                        help='Backbone 类型')
    parser.add_argument('--data_root', type=str,
                        default='orignal_data/Amur Tiger Re-identification',
                        help='ATRW 数据根目录')
    parser.add_argument('--eval_script_dir', type=str, default='ATRWEvalScript-main',
                        help='ATRWEvalScript 目录')
    parser.add_argument('--output_dir', type=str, default='outputs/atrw_eval',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='Closed-Set 划分的随机种子')
    parser.add_argument('--skip_closedset', action='store_true',
                        help='跳过 Closed-Set 评估')
    parser.add_argument('--skip_openset', action='store_true',
                        help='跳过 Open-Set 评估')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "="*70)
    print("ATRW 双协议评估")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backbone: {args.backbone}")
    print(f"Timestamp: {timestamp}")
    print("="*70)
    
    results = {}
    
    # 1. Closed-Set 评估
    if not args.skip_closedset:
        print("\n" + "#"*70)
        print("# 协议 1: Closed-Set (7:3 Split)")
        print("# 与 SMFFEN 2024 等论文使用相同协议")
        print("#"*70)
        
        cmd = f'''python tools/eval_atrw_closedset.py \
            --checkpoint "{args.checkpoint}" \
            --backbone {args.backbone} \
            --data_root "{args.data_root}" \
            --seed {args.seed}'''
        
        success = run_command(cmd, "运行 Closed-Set 评估")
        results['closedset'] = 'Success' if success else 'Failed'
    
    # 2. Open-Set 评估
    if not args.skip_openset:
        print("\n" + "#"*70)
        print("# 协议 2: Open-Set (官方协议)")
        print("# 测试集包含训练时未见过的 75 个身份")
        print("#"*70)
        
        output_json = os.path.join(args.output_dir, f'submission_openset_{timestamp}.json')
        
        cmd = f'''python tools/eval_atrw_openset.py \
            --checkpoint "{args.checkpoint}" \
            --backbone {args.backbone} \
            --data_root "{args.data_root}" \
            --eval_script_dir "{args.eval_script_dir}" \
            --output "{output_json}"'''
        
        success = run_command(cmd, "运行 Open-Set 评估")
        results['openset'] = 'Success' if success else 'Failed'
    
    # 总结
    print("\n" + "="*70)
    print("评估完成总结")
    print("="*70)
    
    for protocol, status in results.items():
        print(f"  {protocol}: {status}")
    
    print("\n" + "-"*70)
    print("论文建议:")
    print("-"*70)
    print("""
在论文中，建议按以下方式报告结果:

1. 主表格使用 Closed-Set 结果，与现有方法对比:

   Table X: Comparison with state-of-the-art on ATRW (Closed-Set)
   ┌─────────────────────┬─────────┬─────────┬─────────┐
   │ Method              │ Rank-1  │ Rank-5  │ mAP     │
   ├─────────────────────┼─────────┼─────────┼─────────┤
   │ ResNet50 [X]        │ 91.70   │ 97.90   │ 68.40   │
   │ PCB [X]             │ 94.70   │ 98.40   │ 71.20   │
   │ SMFFEN [X]          │ 96.30   │ 98.90   │ 78.70   │
   │ Ours (IICL)         │ XX.XX   │ XX.XX   │ XX.XX   │
   └─────────────────────┴─────────┴─────────┴─────────┘

2. 补充材料或附加实验使用 Open-Set 结果，展示泛化能力:

   Table Y: Open-Set Evaluation (Official ATRW Protocol)
   ┌─────────────────────┬─────────────┬─────────────┐
   │ Scenario            │ Rank-1      │ mAP         │
   ├─────────────────────┼─────────────┼─────────────┤
   │ Single-camera       │ XX.XX       │ XX.XX       │
   │ Cross-camera        │ XX.XX       │ XX.XX       │
   └─────────────────────┴─────────────┴─────────────┘

   * 注: Open-Set 协议使用 75 个训练时未见过的身份进行测试，
         难度远高于 Closed-Set，结果不可直接与上表对比。
""")
    print("="*70)


if __name__ == '__main__':
    main()
