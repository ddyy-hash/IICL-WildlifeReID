#!/usr/bin/env python3
"""
Illumination Configuration Loader
加载光照模块配置文件
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为 app/core/illumination_config.yaml
        
    Returns:
        配置字典
    """
    if config_path is None:
        # 默认配置路径
        config_path = os.path.join(
            os.path.dirname(__file__), 
            'illumination_config.yaml'
        )
    
    if not os.path.exists(config_path):
        print(f"[WARN] Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Config loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'illumination_module': {
            'module_type': 'IlluminationInvariantModule',
            'loss_params': {
                'lambda_sparse': 0.1,
                'lambda_smooth': 0.2,
                'lambda_orthogonal': 0.1,
            },
            'training': {
                'epochs_per_phase': 20,
                'illumination_lr': 0.0001,
                'reid_lr': 0.0001,
                'detection_weight': 1.0,
                'illumination_weight': 0.8,
            },
        },
        'training': {
            'epochs': 60,
            'batch_size': 8,
            'image_size': 256,
            'data_dir': './data/dogs',
            'metric_learning': {
                'enabled': True,
                'loss_type': 'triplet',
                'loss_params': {
                    'triplet_margin': 0.3,
                    'triplet_mining': 'soft',
                },
            },
        },
        'hardware': {
            'device': 'cuda',
            'workers': 2,
        },
    }


def save_config(config: Dict[str, Any], config_path: str):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 保存路径
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"[OK] Config saved to: {config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")


if __name__ == '__main__':
    # 测试配置加载
    config = load_config()
    print("Loaded config sections:", list(config.keys()))
