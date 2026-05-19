#!/usr/bin/env python3
"""

    python tools/train_stripespotter.py
    
    python tools/train_joint.py --config config/illumination_config_stripespotter.yaml
"""

import os
import sys
import subprocess

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    config_path = "config/illumination_config_stripespotter.yaml"
    
    train_dir = "data/processed/stripespotter/train"
    if not os.path.exists(train_dir):
        print("=" * 60)
        print("ERROR: Training data not found!")
        print("=" * 60)
        print(f"Expected: {train_dir}")
        print("\nPlease run preprocessing first:")
        print("  python tools/preprocess_stripespotter.py")
        sys.exit(1)
    
    n_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    n_images = sum(
        len([f for f in os.listdir(os.path.join(train_dir, d)) if f.endswith('.jpg')])
        for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )
    
    print("=" * 60)
    print("StripeSpotter (Zebra) Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Train data: {train_dir}")
    print(f"  - Classes: {n_classes}")
    print(f"  - Images: {n_images}")
    print("=" * 60)
    print()
    
    cmd = [
        sys.executable,
        "tools/train_joint.py",
        "--config", config_path,
        "--data_dir", "data/processed/stripespotter/train",
        "--query_dir", "data/processed/stripespotter/query",
        "--gallery_dir", "data/processed/stripespotter/gallery",
        "--output_dir", "checkpoints/joint_stripespotter",
        "--num_classes", str(n_classes),
        "--use_iicl",
        "--iicl_weight", "1.5",
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
