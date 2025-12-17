#!/usr/bin/env python3
"""
官方 ATRW Open-Set ReID 评估脚本
使用 ATRWEvalScript 进行标准评估

评估协议说明：
- 这是 OPEN-SET 评估：训练集 107 个 ID，测试集 75 个不同的 ID
- 测试集有 1764 张图片
- 每张图作为 query，需要对所有其他图进行相似度排序
- 结果分为 single-camera 和 cross-camera 两种场景

与 Closed-Set (7:3 split) 的区别：
- Open-Set: 测试时遇到的是训练时从未见过的身份
- 更能反映真实应用场景的泛化能力
- 难度更高，结果通常比 Closed-Set 低很多

使用方法:
    python tools/eval_atrw_openset.py \
        --checkpoint checkpoints/your_model.pth \
        --backbone osnet_x1_0
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import average_precision_score

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.joint_model import JointReIDModel


def evaluate_openset(gt_file, submission):
    """
    内置的 ATRW Open-Set 评估函数
    改编自官方 ATRWEvalScript/atrwtool/plain.py
    """
    with open(gt_file, 'r') as f:
        anno = json.load(f)
    
    # 解析 ground truth
    eids, fids = {}, {}  # entityid, frame info
    query_multi = []  # 跨摄像机 query
    query_sing = []   # 同摄像机 query
    
    for obj in anno:
        imgid = obj['imgid']
        query = obj['query']
        frame = obj['frame']
        entityid = obj['entityid']
        
        if query == 'multi':
            query_multi.append(imgid)
        elif query == 'sing':
            query_sing.append(imgid)
        
        eids[imgid] = entityid
        fids[imgid] = tuple(frame)
    
    def exclude_mask(eid, fid, ans_eids, ans_fids):
        """排除同帧同ID的图片（junk images）"""
        ans_eids = np.asarray(ans_eids)
        ans_fids = np.asarray(ans_fids)
        
        eid_match = (ans_eids == eid)
        cid_match = np.logical_and(fid[0] == ans_fids[:, 0], fid[1] == ans_fids[:, 1])
        frame_match = np.abs(ans_fids[:, 2] - fid[2]) <= 3
        
        mask = np.logical_and(cid_match, eid_match)
        mask = np.logical_and(mask, frame_match)
        junk_images = ans_eids == -1
        mask = np.logical_or(mask, junk_images)
        
        return mask
    
    # 评估
    aps_sing, aps_multi = [], []
    cmc_sing = np.zeros(len(eids), dtype=np.int32)
    cmc_multi = np.zeros(len(eids), dtype=np.int32)
    
    for ans in submission:
        query_id = ans['query_id']
        ans_ids = ans['ans_ids']
        
        if query_id in query_multi:
            aps = aps_multi
            cmc = cmc_multi
        elif query_id in query_sing:
            aps = aps_sing
            cmc = cmc_sing
        else:
            continue
        
        entityid = eids[query_id]
        gt_eids = np.asarray([eids[i] for i in ans_ids])
        pid_matches = gt_eids == entityid
        
        # 排除 junk images
        mask = exclude_mask(entityid, fids[query_id], gt_eids, [fids[i] for i in ans_ids])
        distances = np.arange(len(ans_ids)) + 1.0
        distances[mask] = np.inf
        pid_matches[mask] = False
        
        scores = 1 / distances
        ap = average_precision_score(pid_matches, scores)
        
        if np.isnan(ap):
            aps.append(0)
            continue
        
        aps.append(ap)
        
        # CMC
        sorted_matches = pid_matches[np.argsort(distances)]
        if sorted_matches.sum() > 0:
            k = np.where(sorted_matches)[0][0]
            cmc[k:] += 1
    
    # 计算指标
    results = {
        'mAP(single_cam)': np.mean(aps_sing) * 100 if aps_sing else 0,
        'top-1(single_cam)': cmc_sing[0] / len(query_sing) * 100 if query_sing else 0,
        'top-5(single_cam)': cmc_sing[4] / len(query_sing) * 100 if query_sing else 0,
        'mAP(cross_cam)': np.mean(aps_multi) * 100 if aps_multi else 0,
        'top-1(cross_cam)': cmc_multi[0] / len(query_multi) * 100 if query_multi else 0,
        'top-5(cross_cam)': cmc_multi[4] / len(query_multi) * 100 if query_multi else 0,
    }
    
    return results


class ATRWTestDataset(Dataset):
    """ATRW 测试集"""
    
    def __init__(self, test_images, transform=None):
        self.test_images = test_images
        self.transform = transform or T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            # 注意: 不做 Normalize，模型内部会自己归一化
        ])
    
    def __len__(self):
        return len(self.test_images)
    
    def __getitem__(self, idx):
        img_info = self.test_images[idx]
        img = Image.open(img_info['path']).convert('RGB')
        img = self.transform(img)
        return img, img_info['imgid']


def get_test_images(test_dir, gt_file=None):
    """
    获取测试集图片列表
    
    注意: ATRW 测试图片命名格式为 000000.jpg, 000004.jpg, ...
    imgid 就是文件名（无扩展名）转为整数
    imgid 不连续，范围是 0-5148，共 1764 张图
    """
    # 如果有 ground truth，获取有效的 imgid 列表
    valid_imgids = None
    if gt_file and os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        valid_imgids = set(obj['imgid'] for obj in gt)
        print(f"Ground truth contains {len(valid_imgids)} images")
    
    test_images = []
    
    # 遍历测试目录
    for img_name in sorted(os.listdir(test_dir)):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(test_dir, img_name)
            # imgid 从文件名提取 (000000.jpg -> 0)
            imgid = int(os.path.splitext(img_name)[0])
            
            # 如果有 ground truth，只加载其中存在的图片
            if valid_imgids is not None and imgid not in valid_imgids:
                continue
            
            test_images.append({
                'imgid': imgid,
                'path': img_path,
                'filename': img_name
            })
    
    return test_images


def extract_features(model, dataloader, device):
    """批量提取特征"""
    model.eval()
    
    all_features = []
    all_imgids = []
    
    with torch.no_grad():
        for images, imgids in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # 获取特征
            outputs = model(images)
            if isinstance(outputs, dict):
                features = outputs.get('features', outputs.get('global_feat'))
            elif isinstance(outputs, (tuple, list)):
                features = outputs[0]
            else:
                features = outputs
            
            features = F.normalize(features, dim=1)
            
            all_features.append(features.cpu())
            all_imgids.extend(imgids.tolist())
    
    features = torch.cat(all_features, dim=0)
    return features, all_imgids


def generate_submission(imgids, features, output_path):
    """
    生成提交文件
    
    对于每张图片作为 query，排序所有其他图片
    """
    num_images = len(imgids)
    
    # 计算距离矩阵 (余弦距离)
    print("Computing distance matrix...")
    similarity = torch.mm(features, features.t())
    distance_matrix = (1 - similarity).numpy()
    
    submission = []
    
    for i in tqdm(range(num_images), desc="Generating submission"):
        query_id = imgids[i]
        distances = distance_matrix[i].copy()
        
        # 排除自己
        distances[i] = float('inf')
        
        # 按距离升序排列
        sorted_indices = np.argsort(distances)
        ans_ids = [imgids[idx] for idx in sorted_indices]
        
        submission.append({
            'query_id': query_id,
            'ans_ids': ans_ids
        })
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"Submission saved to {output_path}")
    return submission


def main():
    parser = argparse.ArgumentParser(description='ATRW Open-Set Official Evaluation')
    parser.add_argument('--test_dir', type=str, 
                        default=None,
                        help='Path to test images directory (auto-detect if not specified)')
    parser.add_argument('--data_root', type=str,
                        default='orignal_data/Amur Tiger Re-identification',
                        help='ATRW data root directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='osnet_ain_x1_0',
                        help='Backbone type')
    parser.add_argument('--num_classes', type=int, default=107,
                        help='Number of training classes (for model init)')
    parser.add_argument('--output', type=str, default='submission_openset.json',
                        help='Output submission file path')
    parser.add_argument('--eval_script_dir', type=str, default='ATRWEvalScript-main',
                        help='Path to ATRWEvalScript directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip running evaluation script')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 检查文件路径
    print("\n" + "="*60)
    print("Step 1: 检查文件路径")
    print("="*60)
    
    # 自动检测测试目录
    if args.test_dir is None:
        # 尝试服务器结构: data_root/test/
        test_dir = os.path.join(args.data_root, 'test')
        if not os.path.exists(test_dir):
            # 尝试本地结构: data_root/atrw_reid_test/test/
            test_dir = os.path.join(args.data_root, 'atrw_reid_test', 'test')
        args.test_dir = test_dir
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    gt_file = os.path.join(args.eval_script_dir, 'annotations', 'gt_test_plain.json')
    if not os.path.exists(gt_file):
        print(f"Warning: Ground truth not found: {gt_file}")
        gt_file = None
    
    print(f"Test dir: {args.test_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Ground truth: {gt_file}")
    
    # 2. 加载测试图片
    print("\n" + "="*60)
    print("Step 2: 加载测试图片")
    print("="*60)
    
    test_images = get_test_images(args.test_dir, gt_file)
    print(f"Loaded {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("Error: No test images found!")
        return
    
    # 3. 创建数据加载器
    test_dataset = ATRWTestDataset(test_images)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 4. 加载模型
    print("\n" + "="*60)
    print("Step 3: 加载模型")
    print("="*60)
    
    model = JointReIDModel(
        num_classes=args.num_classes,
        backbone_name=args.backbone,
        pretrained_backbone=False  # 不下载预训练，直接加载 checkpoint
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 处理可能的 key 不匹配
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() 
                       if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} parameters")
    
    model = model.to(device)
    model.eval()
    
    # 5. 提取特征
    print("\n" + "="*60)
    print("Step 4: 提取特征")
    print("="*60)
    
    features, imgids = extract_features(model, test_loader, device)
    print(f"Feature shape: {features.shape}")
    
    # 6. 生成提交文件
    print("\n" + "="*60)
    print("Step 5: 生成提交文件")
    print("="*60)
    
    generate_submission(imgids, features, args.output)
    
    # 7. 运行评估
    if not args.skip_eval and gt_file:
        print("\n" + "="*60)
        print("Step 6: 运行 Open-Set 评估")
        print("="*60)
        
        # 加载 submission
        with open(args.output, 'r') as f:
            submission = json.load(f)
        
        # 使用内置评估函数
        results = evaluate_openset(gt_file, submission)
        
        print("\n" + "="*60)
        print("Open-Set 评估结果 (官方 ATRW 协议)")
        print("="*60)
        print(f"\n{'场景':<20} {'Rank-1':<12} {'Rank-5':<12} {'mAP':<12}")
        print("-"*60)
        print(f"{'Single-camera':<20} {results['top-1(single_cam)']:>6.2f}%      {results['top-5(single_cam)']:>6.2f}%      {results['mAP(single_cam)']:>6.2f}%")
        print(f"{'Cross-camera':<20} {results['top-1(cross_cam)']:>6.2f}%      {results['top-5(cross_cam)']:>6.2f}%      {results['mAP(cross_cam)']:>6.2f}%")
        print("-"*60)
        
        print("\n" + "-"*60)
        print("Open-Set 评估说明:")
        print("-"*60)
        print("- Single-camera: 同一摄像机下的重识别 (相对容易)")
        print("- Cross-camera: 跨摄像机重识别 (更具挑战性)")
        print("- 测试集包含 75 个训练时从未见过的老虎身份")
        print("- 这是官方 ATRW 竞赛评估协议")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)


if __name__ == '__main__':
    main()
