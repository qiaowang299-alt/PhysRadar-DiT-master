"""
RD数据集模块
实现PyTorch Dataset用于读取和处理RD图数据
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Union
import cv2
import logging


class RDDataset(Dataset):
    """
    RD数据集类 - 从.npy文件读取RD图，输出适合DiT或ControlNet训练的格式
    """
    
    def __init__(self, 
                 data_dir: str,
                 target_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True,
                 transform: Optional[callable] = None,
                 file_pattern: str = "*.npy"):
        """
        初始化RD数据集
        
        Args:
            data_dir: 数据目录路径，包含.npy文件
            target_size: 目标尺寸 (height, width)
            normalize: 是否归一化数据到[0,1]
            transform: 数据增强变换
            file_pattern: 文件匹配模式
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.normalize = normalize
        self.transform = transform
        
        # 查找所有.npy文件
        self.file_paths = list(self.data_dir.rglob(file_pattern))
        
        if not self.file_paths:
            raise ValueError(f"在目录 {data_dir} 中未找到匹配 {file_pattern} 的文件")
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        print(f"RD数据集初始化完成:")
        print(f"  数据目录: {self.data_dir}")
        print(f"  文件数量: {len(self.file_paths)}")
        print(f"  目标尺寸: {target_size}")
        print(f"  归一化: {normalize}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            rd_tensor: RD图张量，形状为 (1, D, R) 的float32张量
        """
        try:
            # 1. 加载.npy文件
            file_path = self.file_paths[idx]
            rd_data = np.load(file_path)
            
            # 2. 验证数据形状
            rd_data = self._validate_and_preprocess(rd_data)
            
            # 3. 转换为PyTorch张量
            rd_tensor = torch.from_numpy(rd_data).float()
            
            # 4. 添加通道维度 (1, D, R)
            if len(rd_tensor.shape) == 2:
                rd_tensor = rd_tensor.unsqueeze(0)  # (1, D, R)
            
            # 5. 调整尺寸到目标大小
            rd_tensor = self._resize_or_pad(rd_tensor)
            
            # 6. 归一化到[0,1]
            if self.normalize:
                rd_tensor = self._normalize(rd_tensor)
            
            # 7. 应用数据增强
            if self.transform:
                rd_tensor = self.transform(rd_tensor)
            
            return rd_tensor
            
        except Exception as e:
            self.logger.error(f"加载文件失败: {self.file_paths[idx]} - {e}")
            # 返回零张量作为错误处理
            return torch.zeros((1, *self.target_size), dtype=torch.float32)
    
    def _validate_and_preprocess(self, rd_data: np.ndarray) -> np.ndarray:
        """
        验证和预处理RD数据
        
        Args:
            rd_data: 原始RD数据
            
        Returns:
            processed_data: 处理后的RD数据
        """
        # 检查数据类型
        if rd_data.dtype not in [np.float32, np.float64]:
            rd_data = rd_data.astype(np.float32)
        
        # 检查维度
        if len(rd_data.shape) == 3:
            # (C, D, R) -> 取第一个通道
            if rd_data.shape[0] > 1:
                self.logger.warning(f"多通道RD数据，取第一个通道: {rd_data.shape}")
            rd_data = rd_data[0]  # (D, R)
        
        elif len(rd_data.shape) == 2:
            # (D, R) - 正确格式
            pass
        else:
            raise ValueError(f"不支持的RD数据形状: {rd_data.shape}")
        
        # 检查数值范围
        if np.isnan(rd_data).any() or np.isinf(rd_data).any():
            self.logger.warning("RD数据包含NaN或Inf值，进行清理")
            rd_data = np.nan_to_num(rd_data)
        
        return rd_data
    
    def _resize_or_pad(self, rd_tensor: torch.Tensor) -> torch.Tensor:
        """
        调整RD图尺寸到目标大小
        
        Args:
            rd_tensor: 输入RD张量，形状为 (1, D, R)
            
        Returns:
            resized_tensor: 调整后的张量，形状为 (1, target_H, target_W)
        """
        _, current_h, current_w = rd_tensor.shape
        
        # 如果当前尺寸与目标尺寸相同，直接返回
        if (current_h, current_w) == self.target_size:
            return rd_tensor
        
        # 转换为numpy进行OpenCV处理
        rd_np = rd_tensor.squeeze(0).numpy()  # (D, R)
        
        # 使用双线性插值调整尺寸
        resized_np = cv2.resize(rd_np, (self.target_size[1], self.target_size[0]), 
                               interpolation=cv2.INTER_LINEAR)
        
        # 转换回PyTorch张量
        resized_tensor = torch.from_numpy(resized_np).unsqueeze(0)  # (1, H, W)
        
        return resized_tensor
    
    def _normalize(self, rd_tensor: torch.Tensor) -> torch.Tensor:
        """
        归一化RD数据到[0,1]范围
        
        Args:
            rd_tensor: 输入RD张量
            
        Returns:
            normalized_tensor: 归一化后的张量
        """
        # 计算最小值和最大值
        min_val = rd_tensor.min()
        max_val = rd_tensor.max()
        
        # 避免除零
        if max_val - min_val > 1e-8:
            normalized = (rd_tensor - min_val) / (max_val - min_val)
        else:
            normalized = rd_tensor - min_val  # 如果所有值相同，归一化到0
        
        return normalized
    
    def get_file_info(self, idx: int) -> dict:
        """
        获取文件信息
        
        Args:
            idx: 文件索引
            
        Returns:
            info: 文件信息字典
        """
        file_path = self.file_paths[idx]
        rd_data = np.load(file_path)
        
        return {
            'file_path': str(file_path),
            'original_shape': rd_data.shape,
            'data_type': rd_data.dtype,
            'data_range': (rd_data.min(), rd_data.max()),
            'file_size': os.path.getsize(file_path)
        }
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple['RDDataset', 'RDDataset', 'RDDataset']:
        """
        分割数据集为训练集、验证集和测试集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            train_set, val_set, test_set: 分割后的数据集
        """
        total_files = len(self.file_paths)
        train_size = int(total_files * train_ratio)
        val_size = int(total_files * val_ratio)
        test_size = total_files - train_size - val_size
        
        # 随机打乱文件顺序
        np.random.seed(42)  # 固定随机种子以保证可重复性
        indices = np.random.permutation(total_files)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 创建子数据集
        train_files = [self.file_paths[i] for i in train_indices]
        val_files = [self.file_paths[i] for i in val_indices]
        test_files = [self.file_paths[i] for i in test_indices]
        
        # 创建新的数据集实例
        train_set = self._create_subset(train_files)
        val_set = self._create_subset(val_files)
        test_set = self._create_subset(test_files)
        
        print(f"数据集分割完成:")
        print(f"  训练集: {len(train_set)} 个样本")
        print(f"  验证集: {len(val_set)} 个样本")
        print(f"  测试集: {len(test_set)} 个样本")
        
        return train_set, val_set, test_set
    
    def _create_subset(self, file_paths: List[Path]) -> 'RDDataset':
        """创建子数据集"""
        subset = RDDataset.__new__(RDDataset)
        subset.data_dir = self.data_dir
        subset.target_size = self.target_size
        subset.normalize = self.normalize
        subset.transform = self.transform
        subset.file_paths = file_paths
        subset.logger = self.logger
        
        return subset


def test_rd_dataset():
    """测试RD数据集功能"""
    import tempfile
    import shutil
    
    print("测试RD数据集...")
    
    # 创建临时目录和测试数据
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_rd_data"
        test_dir.mkdir(parents=True)
        
        # 创建一些测试.npy文件
        num_files = 10
        target_size = (64, 64)
        
        for i in range(num_files):
            # 创建随机RD数据 (D, R)
            rd_data = np.random.randn(128, 256).astype(np.float32)
            file_path = test_dir / f"rd_data_{i:03d}.npy"
            np.save(file_path, rd_data)
        
        # 创建数据集
        dataset = RDDataset(str(test_dir), target_size=target_size)
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试单个样本
        sample = dataset[0]
        print(f"样本形状: {sample.shape}")
        print(f"样本数据类型: {sample.dtype}")
        print(f"样本数值范围: [{sample.min():.3f}, {sample.max():.3f}]")
        
        # 测试文件信息
        file_info = dataset.get_file_info(0)
        print(f"文件信息: {file_info}")
        
        # 测试数据集分割
        train_set, val_set, test_set = dataset.split_dataset()
        print(f"分割结果 - 训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")
        
        print("✓ RD数据集测试完成")


if __name__ == "__main__":
    test_rd_dataset()