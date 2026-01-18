"""
ColoRadar数据集类
用于PhysRadar-DiT训练
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from .colo_radar_processor import ColoRadarProcessor


class ColoRadarDataset(Dataset):
    """ColoRadar数据集类"""
    
    def __init__(self, 
                 data_path: str,
                 sequence_names: List[str],
                 input_size: Tuple[int, int] = (256, 256),
                 transform=None,
                 use_bev: bool = True,
                 max_frames_per_sequence: int = 100):
        """
        初始化数据集
        
        Args:
            data_path: 数据根目录
            sequence_names: 序列名称列表
            input_size: 输入尺寸 (H, W)
            transform: 数据增强变换
            use_bev: 是否使用BEV布局
            max_frames_per_sequence: 每个序列最大帧数
        """
        self.data_path = Path(data_path)
        self.sequence_names = sequence_names
        self.input_size = input_size
        self.transform = transform
        self.use_bev = use_bev
        self.max_frames = max_frames_per_sequence
        
        # 数据处理器
        self.processor = ColoRadarProcessor(data_path)
        
        # 加载所有数据
        self.samples = self._load_all_samples()
        
    def _load_all_samples(self) -> List[Dict]:
        """加载所有样本"""
        samples = []
        
        for seq_name in self.sequence_names:
            print(f"加载序列: {seq_name}")
            
            try:
                # 处理序列数据
                processed_data = self.processor.process_sequence(
                    seq_name, max_frames=self.max_frames
                )
                
                # 添加到样本列表
                for i in range(len(processed_data['radar_maps'])):
                    sample = {
                        'radar_map': processed_data['radar_maps'][i],
                        'bev_layout': processed_data['bev_layouts'][i] if self.use_bev else None,
                        'sequence': seq_name,
                        'frame_idx': processed_data['frame_indices'][i]
                    }
                    samples.append(sample)
                    
            except Exception as e:
                print(f"加载序列 {seq_name} 失败: {e}")
                continue
        
        print(f"总共加载 {len(samples)} 个样本")
        return samples
    
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
            # 如果没有BEV数据，生成随机BEV
            bev_layout = torch.randn(3, self.input_size[0], self.input_size[1])
        
        # 数据增强
        if self.transform:
            radar_map = self.transform(radar_map)
            if self.use_bev:
                bev_layout = self.transform(bev_layout)
        
        return {
            'radar': radar_map,
            'bev': bev_layout if self.use_bev else None,
            'sequence': sample['sequence'],
            'frame_idx': sample['frame_idx']
        }
    
    def _resize_radar(self, radar_map: torch.Tensor) -> torch.Tensor:
        """调整雷达数据尺寸"""
        if radar_map.shape[1:] == self.input_size:
            return radar_map
        
        # 使用插值调整尺寸
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
        
        # 使用插值调整尺寸
        bev_layout = torch.nn.functional.interpolate(
            bev_layout.unsqueeze(0), 
            size=self.input_size, 
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return bev_layout


def get_colo_radar_dataloader(
    data_path: str,
    batch_size: int = 4,
    input_size: Tuple[int, int] = (128, 128),
    num_workers: int = 4,
    use_bev: bool = True
) -> DataLoader:
    """
    获取ColoRadar数据加载器
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        input_size: 输入尺寸
        num_workers: 工作进程数
        use_bev: 是否使用BEV
        
    Returns:
        数据加载器
    """
    # 使用实际的数据集路径和序列名称
    test_sequences = [
        "2_28_2021_outdoors_run9"  # 您的实际序列
    ]
    
    dataset = ColoRadarDataset(
        data_path=data_path,
        sequence_names=test_sequences,
        input_size=input_size,
        use_bev=use_bev,
        max_frames_per_sequence=50  # 每个序列取50帧测试
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# 测试代码
if __name__ == "__main__":
    # 测试数据加载器 - 使用您的实际路径
    dataloader = get_colo_radar_dataloader(
        data_path="D:/桌面/PhysRadar-DiT-master/RaDar/coloradar/dataset",  # 修改为您的实际路径
        batch_size=2,
        input_size=(128, 128)
    )
    
    # 测试一个批次
    for batch_idx, batch in enumerate(dataloader):
        print(f"批次 {batch_idx}:")
        print(f"  雷达数据形状: {batch['radar'].shape}")
        if batch['bev'] is not None:
            print(f"  BEV数据形状: {batch['bev'].shape}")
        print(f"  序列: {batch['sequence']}")
        print(f"  帧索引: {batch['frame_idx']}")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break