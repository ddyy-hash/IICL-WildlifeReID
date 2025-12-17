# IICL-WildlifeReID

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ATRW SOTA](https://img.shields.io/badge/ATRW-97.88%25%20Rank--1-brightgreen.svg)](https://github.com/ddyy-hash/IICL-WildlifeReID)
[![Datasets](https://img.shields.io/badge/Datasets-8%2B-blue.svg)](https://github.com/ddyy-hash/IICL-WildlifeReID)

**Illumination-Invariant Contrastive Learning for Wildlife Re-Identification**

A state-of-the-art wildlife re-identification system featuring **IPAID** (Identity-Preserving Adaptive Illumination Decomposition) and **IICL** (Illumination-Invariant Contrastive Learning) for robust animal identification under varying lighting conditions.

## Highlights

- **State-of-the-Art Performance**: Achieves **97.88% Rank-1** on ATRW (Amur Tiger), surpassing previous SOTA by +1.58%
- **Illumination Robustness**: Physics-based Retinex decomposition handles extreme lighting variations in wildlife imagery
- **Multi-Species Support**: Validated on 8+ wildlife datasets including tigers, zebras, giraffes, pandas, leopards, and lynx
- **End-to-End Training**: Two-phase training strategy with contrastive learning for enhanced generalization
- **Web Interface**: Flask-based GUI for real-time wildlife identification

## Performance

### ATRW (Amur Tiger) - Closed-Set Protocol

| Method | Backbone | Rank-1 | Rank-5 | Rank-10 | mAP |
|--------|----------|--------|--------|---------|-----|
| ResNet50 | ResNet-50 | 91.70% | 97.90% | - | 68.40% |
| PCB (ECCV'18) | ResNet-50 | 94.70% | 98.40% | - | 71.20% |
| SMFFEN (2024) | ResNet-50 | 96.30% | 98.90% | - | 78.70% |
| **Ours (IPAID+IICL)** | OSNet-AIN | **97.88%** | **99.35%** | **99.51%** | **79.15%** |

### ATRW - Open-Set Protocol (Unseen Identities)

| Method | Single-cam Rank-1 | Single-cam mAP | Cross-cam Rank-1 | Cross-cam mAP |
|--------|-------------------|----------------|------------------|---------------|
| APR (ATRW 2020) | 72.5% | 60.1% | 55.3% | 31.8% |
| **Ours (IPAID+IICL)** | **86.02%** | **64.71%** | **72.34%** | **37.63%** |

### Cross-Dataset Evaluation

| Dataset | Species | #IDs | #Images | Rank-1 | mAP |
|---------|---------|------|---------|--------|-----|
| ATRW (Closed) | Amur Tiger | 107 | 1,887 | **97.88%** | **79.15%** |
| ATRW (Open-Single) | Amur Tiger | 182 | 1,764 | **86.02%** | **64.71%** |
| GZGC-Zebra | Grevy's Zebra | 1,033 | 3,851 | **71.16%** | **68.77%** |
| GZGC-Giraffe | Masai Giraffe | 109 | 420 | **62.60%** | **61.16%** |
| StripeSpotter | Plains Zebra | 44 | 565 | **96.00%** | **93.20%** |

## Technical Innovation

### IPAID: Identity-Preserving Adaptive Illumination Decomposition

Based on Retinex theory, IPAID performs physics-based illumination decomposition:

```
I = R × L

Where:
  I = Input image (observed)
  R = Reflectance (intrinsic texture, identity-preserving)
  L = Illumination (lighting conditions, to be normalized)
```

**Key Features:**
- Multi-scale illumination estimation (global + local)
- Sensitivity-guided adaptive correction
- Identity-preserving constraints on reflectance layer
- Plug-and-play design compatible with any backbone

### IICL: Illumination-Invariant Contrastive Learning

A novel contrastive learning approach that generates positive pairs via illumination transformation:

```
Original Image I = R × L
                    │
                    ├── R (Reflectance) ← Identity features preserved
                    │
                    ├── L → L' (Illumination transform: gamma/scale/local perturbation)
                    │
                    └── I' = R × L' (Illumination variant)

Contrastive Loss: f(I) ≈ f(I')  (Same identity, different lighting → similar features)
```

**Advantages over traditional augmentation:**
1. Preserves texture features (R unchanged) vs. color jitter may destroy patterns
2. Physically plausible data augmentation (1 image → N lighting variants)
3. No need for real illumination paired data
4. End-to-end differentiable training

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Image                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    IPAID Module                                      │
│  ┌──────────────────┐    ┌──────────────────┐                       │
│  │ Multi-scale      │    │ Sensitivity      │                       │
│  │ Illumination Est.│───▶│ Map Generator    │                       │
│  └──────────────────┘    └──────────────────┘                       │
│           │                      │                                   │
│           ▼                      ▼                                   │
│  ┌──────────────────┐    ┌──────────────────┐                       │
│  │ L (Illumination) │    │ Adaptive         │                       │
│  │                  │───▶│ Correction       │──▶ R (Reflectance)    │
│  └──────────────────┘    └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OSNet-AIN Backbone                                │
│  • Omni-Scale Feature Aggregation                                   │
│  • Adaptive Instance Normalization                                   │
│  • 512-dim Feature Embedding                                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Local Feature Extractor                           │
│  • Part-based pooling (6 parts)                                      │
│  • SE Attention mechanism                                            │
│  • GeM Pooling                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    256-dim Feature Vector
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/ddyy-hash/IICL-WildlifeReID.git
cd IICL-WildlifeReID

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install PyTorch (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
torchreid
numpy
opencv-python
Pillow
PyYAML
tqdm
matplotlib
seaborn
ultralytics
gdown
Flask
WTForms
celery
```

## Usage

### 1. Data Preparation

#### Download Datasets

| Dataset | Species | Download Link |
|---------|---------|---------------|
| ATRW | Amur Tiger | [Official Site](https://cvwc2019.github.io/challenge.html) |
| GZGC | Zebra & Giraffe | [Lila Science](https://lila.science/datasets/great-zebra-giraffe-id) |
| StripeSpotter | Plains Zebra | [Google Code Archive](https://code.google.com/archive/p/stripespotter/downloads) |
| iPanda50 | Giant Panda | [GitHub](https://github.com/iPandaDateset/iPanda-50) |
| CzechLynx | Eurasian Lynx | [Kaggle](https://www.kaggle.com/datasets/picekl/czechlynx/data) |

#### Preprocess Data

```bash
# ATRW (Amur Tiger)
python tools/preprocess_data.py --dataset atrw --src_dir orignal_data/ATRW --dst_dir data/processed/atrw

# GZGC (Zebra & Giraffe)
python tools/preprocess_gzgc.py --src_dir orignal_data/gzgc.coco --dst_dir data/processed/gzgc

# StripeSpotter (Plains Zebra)
python tools/preprocess_stripespotter.py --src_dir orignal_data/StripeSpotter --dst_dir data/processed/stripespotter

# Leopard
python tools/preprocess_leopard_v2.py --src_dir orignal_data/leopard.coco --dst_dir data/processed/leopard
```

### 2. Training

#### Basic Training

```bash
python tools/train_joint.py \
    --config app/core/illumination_config.yaml \
    --data_dir data/processed/atrw/train \
    --output_dir checkpoints/atrw_ipaid_iicl \
    --query_dir data/processed/atrw/query \
    --gallery_dir data/processed/atrw/gallery \
    --use_iicl --iicl_weight 1.0 --iicl_variants 2
```

#### Training with Custom Configuration

```bash
# For GZGC Giraffe
python tools/train_joint.py \
    --config app/core/illumination_config_gzgc_giraffe.yaml \
    --data_dir data/processed/gzgc_giraffe/train \
    --output_dir checkpoints/joint_gzgc_giraffe \
    --query_dir data/processed/gzgc_giraffe/query \
    --gallery_dir data/processed/gzgc_giraffe/gallery \
    --phase2_epochs 80 \
    --use_iicl --iicl_weight 1.0 --iicl_variants 2
```

### 3. Evaluation

```bash
# Standard evaluation
python tools/evaluate_reid.py \
    --checkpoint checkpoints/atrw_ipaid_iicl/joint_best_reid_best.pth \
    --query_dir data/processed/atrw/query \
    --gallery_dir data/processed/atrw/gallery

# ATRW Open-Set evaluation
python tools/eval_atrw_openset.py \
    --checkpoint checkpoints/atrw_ipaid_iicl/joint_best_reid_best.pth \
    --test_dir data/processed/atrw/test
```

## Configuration

### Training Configuration (YAML)

```yaml
# Example: illumination_config.yaml
illumination_module:
  module_type: "IPAIDModule"
  loss_params:
    lambda_recon: 1.0      # Reconstruction loss
    lambda_smooth: 0.15    # Illumination smoothness
    lambda_edge: 0.08      # Edge preservation
    lambda_identity: 0.1   # Identity preservation
  module_params:
    base_channels: 32      # Lightweight design
    num_scales: 3          # Multi-scale estimation
    refine_iterations: 2   # Refinement iterations

training:
  batch_size: 32           # P=8, K=4
  learning_rate: 0.00035
  weight_decay: 0.01       # Strong regularization
  
  phases:
    phase1:                # Backbone warmup
      epochs: 10
      freeze_illumination: true
    phase2:                # Joint training
      epochs: 100
      illumination_weight: 0.2
      
  losses:
    cross_entropy:
      weight: 1.0
      label_smoothing: 0.2
    arcface:
      enabled: true
      weight: 0.5
    triplet:
      weight: 1.0
      margin: 0.3
    iicl:
      enabled: true
      weight: 1.0
```

### Key Hyperparameters

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| `learning_rate` | 3.5e-4 | Base learning rate |
| `weight_decay` | 0.01 | L2 regularization |
| `label_smoothing` | 0.2 | Prevents overconfidence |
| `iicl_weight` | 1.0 | IICL contrastive loss weight |
| `illumination_weight` | 0.2 | Illumination regularization |
| `phase1_epochs` | 10 | Backbone warmup |
| `phase2_epochs` | 100 | Joint training |

## Two-Phase Training Strategy

```
Phase 1: Backbone Warmup (10 epochs)
├── Freeze: IPAID illumination module
├── Train: OSNet-AIN backbone + Local extractor
├── Loss: L_reid only (CE + Triplet + ArcFace)
└── Purpose: Establish stable feature representations

Phase 2: Joint Training (100 epochs)
├── Unfreeze: All modules
├── Train: End-to-end optimization
├── Loss: L_reid + 0.2 × L_illum + 1.0 × L_IICL
└── Purpose: Learn illumination-invariant features
```

## Supported Backbones

| Backbone | Params | Feature Dim | Recommended For |
|----------|--------|-------------|-----------------|
| `osnet_ain_x1_0` | 4.5M | 512 | **Default (Best)** |
| `osnet_x1_0` | 2.2M | 512 | Lightweight |
| `resnet50` | 25.6M | 2048 | Comparison |
| `resnet101` | 44.5M | 2048 | High capacity |

## Supported Datasets

| Dataset | Species | Identities | Images | Challenge |
|---------|---------|------------|--------|-----------|
| ATRW | Amur Tiger | 107/182 | 1,887 | Stripes, pose variation |
| GZGC-Zebra | Grevy's Zebra | 1,033 | 3,851 | Large-scale, lighting |
| GZGC-Giraffe | Masai Giraffe | 109 | 420 | Sparse samples |
| StripeSpotter | Plains Zebra | 44 | 565 | Clean patterns |
| iPanda50 | Giant Panda | 50 | ~2,000 | Similar appearance |
| CzechLynx | Eurasian Lynx | 319 | ~3,500 | Synthetic + Real |
| Leopard | African Leopard | ~200 | ~1,500 | Spots, occlusion |
| Nyala | Nyala Antelope | ~150 | ~1,200 | Stripes, lighting |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{ding2025ipaid,
  title={IPAID-IICL: Illumination-Invariant Contrastive Learning for Wildlife Re-Identification},
  author={Ding, Yu},
  year={2025},
  howpublished={\url{https://github.com/ddyy-hash/IICL-WildlifeReID}},
  note={GitHub repository}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [torchreid](https://github.com/KaiyangZhou/deep-person-reid) - Deep learning person re-identification library
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid) - Omni-Scale Feature Learning
- [ATRW](https://cvwc2019.github.io/challenge.html) - Amur Tiger Re-identification in the Wild
- [Lila Science](https://lila.science/) - Wildlife image datasets

## Contact

For questions or collaboration, please open an issue or contact:
- **Author**: Ding Yu
- **GitHub**: [https://github.com/ddyy-hash/IICL-WildlifeReID](https://github.com/ddyy-hash/IICL-WildlifeReID)

---

<p align="center">
  <b>Protecting Wildlife Through AI</b>
</p>
