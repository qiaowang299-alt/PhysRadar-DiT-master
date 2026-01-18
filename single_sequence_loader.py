"""
单条ColoRadar序列数据加载器
专门针对 2_28_2021_outdoors_run9 优化
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import random
from .colo_radar_processor import ColoRadarProcessor


class SingleSequenceDataset(Dataset):
    """单条序列数据集类"""
    
    def __init__(self, 
                 data_path: str,
                 sequence_name: str,
                 input_size: Tuple[int, int] = (128, 128),
                 split: str = "train",  # train/val/test
                 transform=None,
                 use_bev: bool = True,
                 max_frames: int = 200):
        """
        初始化单条序列数据集
        
        Args:
            data_path: 数据根目录
            sequence_name: 序列名称 (2_28_2021_outdoors_run9)
            input_size: 输入尺寸
            split: 数据分割
            transform: 数据增强变换
            use_bev: 是否使用BEV布局
            max_frames: 最大使用帧数
        """
        self.data_path = Path(data_path)
        self.sequence_name = sequence_name
        self.input_size = input_size
        self.split = split
        self.transform = transform
        self.use_bev = use_bev
        self.max_frames = max_frames
        
        # 数据处理器
        self.processor = ColoRadarProcessor(data_path)
        
        # 加载和处理数据
        self.samples = self._load_and_split_sequence()
        
        print(f"单条序列 {sequence_name} - {split}分割:")
        print(f"  总帧数: {len(self.samples)}")
        print(f"  输入尺寸: {input_size}")
        
    def _load_and_split_sequence(self) -> List[Dict]:
        """加载并分割单条序列数据"""
        # 处理序列数据
        processed_data = self.processor.process_sequence(
            self.sequence_name, max_frames=self.max_frames
        )
        
        total_frames = len(processed_data['radar_maps'])
        
        # 计算分割点
        train_end = int(total_frames * 0.8)
        val_end = train_end + int(total_frames * 0.15)
        
        # 根据分割选择数据
        if self.split == "train":
            indices = range(0, train_end)
        elif self.split == "val":
            indices = range(train_end, val_end)
        elif self.split == "test":
            indices = range(val_end, total_frames)
        else:
            indices = range(total_frames)
        
        samples = []
        for idx in indices:
            sample = {
                'radar_map': processed_data['radar_maps'][idx],
                'bev_layout': processed_data['bev_layouts'][idx] if self.use_bev else None,
                'sequence': self.sequence_name,
                'frame_idx': processed_data['frame_indices'][idx],
                'split': self.split
            }
            samples.append(sample)
        
        # 对训练集进行数据增强
        if self.split == "train":
            samples = self._augment_training_data(samples)
        
        return samples
    
    def _augment_training_data(self, samples: List[Dict]) -> List[Dict]:
        """对训练数据进行增强"""
        augmented_samples = samples.copy()
        
        # 1. 时间序列增强 - 创建连续帧对
        if len(samples) > 10:
            for i in range(len(samples) - 1):
                # 创建连续帧对用于时序学习
                if random.random() < 0.3:  # 30%的概率创建时序样本
                    frame_pair = {
                        'radar_map': samples[i]['radar_map'],
                        'next_radar_map': samples[i+1]['radar_map'],
                        'bev_layout': samples[i]['bev_layout'],
                        'next_bev_layout': samples[i+1]['bev_layout'],
                        'sequence': samples[i]['sequence'],
                        'frame_idx': samples[i]['frame_idx'],
                        'next_frame_idx': samples[i+1]['frame_idx'],
                        'split': 'train_temporal'
                    }
                    augmented_samples.append(frame_pair)
        
        # 2. 数据增强变换
        if self.transform:
            for i in range(len(augmented_samples)):
                augmented_samples[i] = self.transform(augmented_samples[i])
        
        return augmented_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 调整尺寸
        radar_map = self._resize_radar(sample['radar_map'])
        
        if self.use_bev and sample['bev_layout'] is not None:
            bev_layout = self._resize_bev(sample['bev_layout'])
        else:
            # 生成模拟BEV
            bev_layout = self._generate_synthetic_bev()
        
        # 应用数据增强
        if self.transform:
            radar_map = self.transform(radar_map)
            if self.use_bev:
                bev_layout = self.transform(bev_layout)
        
        result = {
            'radar': radar_map,
            'bev': bev_layout if self.use_bev else None,
            'sequence': sample['sequence'],
            'frame_idx': sample['frame_idx'],
            'split': sample['split']
        }
        
        # 如果是时序样本，添加下一帧数据
        if 'next_radar_map' in sample:
            next_radar = self._resize_radar(sample['next_radar_map'])
            next_bev = self._resize_bev(sample['next_bev_layout']) if self.use_bev else None
            
            if self.transform:
                next_radar = self.transform(next_radar)
                if next_bev is not None:
                    next_bev = self.transform(next_bev)
            
            result.update({
                'next_radar': next_radar,
                'next_bev': next_bev,
                'next_frame_idx': sample['next_frame_idx']
            })
        
        return result
    
    def _resize_radar(self, radar_map: torch.Tensor) -> torch.Tensor:
        """调整雷达数据尺寸"""
        if radar_map.shape[1:] == self.input_size:
            return radar_map
        
        radar_map = torch.nn.functional.interpolate(
            radar_map.unsqueeze(0), 
            size=self.input_size, 
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return radar_map
    
    def _resize_bev(self, bev_layout: torch.Tensor) -> torch.Tensor:
        """调整BEV数据尺寸"""
        if bev_layout.shape[1:] == self.input_size:
            return bev_layout
        
        bev_layout = torch.nn.functional.interpolate(
            bev_layout.unsqueeze(0), 
            size=self.input_size, 
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return bev_layout
    
    def _generate_synthetic_bev(self) -> torch.Tensor:
        """生成合成BEV布局 - 针对室外场景"""
        H, W = self.input_size
        
        # 创建室外场景的BEV布局
        bev = torch.zeros(3, H, W)
        
        # 模拟道路和障碍物
        road_width = int(W * 0.3)  # 道路宽度
        road_center = W // 2
        
        # 道路区域（灰色）
        road_start = max(0, road_center - road_width//2)
        road_end = min(W, road_center + road_width//2)
        bev[0, :, road_start:road_end] = 0.5  # 红色通道
        bev[1, :, road_start:road_end] = 0.5  # 绿色通道
        bev[2, :, road_start:road_end] = 0.5  # 蓝色通道
        
        # 随机添加障碍物（白色）
        num_obstacles = random.randint(1, 5)
        for _ in range(num_obstacles):
            obs_h = random.randint(10, H//4)
            obs_w = random.randint(10, W//8)
            obs_x = random.randint(0, W - obs_w)
            obs_y = random.randint(0, H - obs_h)
            
            bev[:, obs_y:obs_y+obs_h, obs_x:obs_x+obs_w] = 1.0
        
        return bev


class OutdoorDataTransform:
    """室外场景数据增强变换"""
    
    def __init__(self, strength: float = 0.3):
        self.strength = strength
    
    def __call__(self, sample: Dict) -> Dict:
        """应用数据增强"""
        radar = sample['radar_map']
        
        # 1. 添加噪声
        if random.random() < 0.5:
            noise_std = self.strength * 0.1
            noise = torch.randn_like(radar) * noise_std
            radar = radar + noise
        
        # 2. 随机裁剪（模拟视角变化）
        if random.random() < 0.3:
            crop_ratio = random.uniform(0.8, 1.0)
            new_h = int(radar.shape[1] * crop_ratio)
            new_w = int(radar.shape[2] * crop_ratio)
            
            start_h = random.randint(0, radar.shape[1] - new_h)
            start_w = random.randint(0, radar.shape[2] - new_w)
            
            radar = radar[:, start_h:start_h+new_h, start_w:start_w+new_w]
            # 重新调整到原尺寸
            radar = torch.nn.functional.interpolate(
                radar.unsqueeze(0), 
                size=(radar.shape[1], radar.shape[2]), 
                mode='bilinear'
            ).squeeze(0)
        
        # 3. 随机翻转（模拟方向变化）
        if random.random() < 0.5:
            radar = torch.flip(radar, dims=[2])  # 水平翻转
        
        sample['radar_map'] = radar
        return sample


def get_single_sequence_dataloaders(
    data_path: str,
    sequence_name: str = "2_28_2021_outdoors_run9",
    batch_size: int = 2,
    input_size: Tuple[int, int] = (128, 128),
    num_workers: int = 2,
    use_bev: bool = True
) -> Dict[str, DataLoader]:
    """
    获取单条序列的数据加载器（训练/验证/测试）
    
    Returns:
        包含三个数据加载器的字典
    """
    # 数据增强变换（仅用于训练集）
    train_transform = OutdoorDataTransform(strength=0.3)
    
    # 创建三个数据集
    datasets = {}
    for split in ['train', 'val', 'test']:
        transform = train_transform if split == 'train' else None
        
        datasets[split] = SingleSequenceDataset(
            data_path=data_path,
            sequence_name=sequence_name,
            input_size=input_size,
            split=split,
            transform=transform,
            use_bev=use_bev,
            max_frames=200  # 最多使用200帧
        )
    
    # 创建数据加载器
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == 'train')  # 仅训练集打乱
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    print(f"单条序列 {sequence_name} 数据加载器创建完成:")
    for split, loader in dataloaders.items():
        print(f"  {split}: {len(loader.dataset)} 样本")
    
    return dataloaders


# 测试代码
if __name__ == "__main__":
    # 测试单条序列数据加载器
    dataloaders = get_single_sequence_dataloaders(
        data_path="D:/ColoRadar/data",  # 修改为实际路径
        sequence_name="2_28_2021_outdoors_run9",
        batch_size=2,
        input_size=(128, 128)
    )
    
    # 测试每个分割
    for split, loader in dataloaders.items():
        print(f"\n{split.upper()} 分割测试:")
        for batch_idx, batch in enumerate(loader):
            print(f"  批次 {batch_idx}:")
            print(f"    雷达数据形状: {batch['radar'].shape}")
            if batch['bev'] is not None:
                print(f"    BEV数据形状: {batch['bev'].shape}")
            print(f"    序列: {batch['sequence']}")
            print(f"    帧索引: {batch['frame_idx']}")
            
            if batch_idx >= 1:  # 只测试前2个批次
                break