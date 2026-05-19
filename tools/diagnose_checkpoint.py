#!/usr/bin/env python3

import torch
import sys

def diagnose_checkpoint(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        print("Using key: model_state_dict")
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
        print("Using key: state_dict")
    else:
        state = ckpt
        print("Using the checkpoint object directly")
    
    print(f"\nTotal parameter entries: {len(state)}")
    
    print("\n=== First 30 parameter names ===")
    for i, k in enumerate(list(state.keys())[:30]):
        print(f"  {k}")
    
    print("\n=== Classifier-related parameters ===")
    for k in state.keys():
        if 'classifier' in k or 'fc' in k.lower():
            print(f"  {k}: {state[k].shape}")
    
    print("\n=== Backbone-related parameters ===")
    backbone_keys = [k for k in state.keys() if 'backbone' in k]
    print(f"Backbone parameter count: {len(backbone_keys)}")
    if backbone_keys:
        print(f"  First key: {backbone_keys[0]}")
    
    if isinstance(ckpt, dict):
        print("\n=== Checkpoint metadata ===")
        for k in ckpt.keys():
            if k not in ['model_state_dict', 'state_dict', 'optimizer_state_dict']:
                print(f"  {k}: {ckpt[k]}")


if __name__ == '__main__':
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/joint_atrw_optimized/joint_best_reid_best.pth'
    diagnose_checkpoint(ckpt_path)
