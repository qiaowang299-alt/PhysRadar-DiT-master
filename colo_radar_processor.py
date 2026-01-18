"""
ColoRadar数据集处理器
将ColoRadar原始数据转换为PhysRadar-DiT训练格式
"""

import os
import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import signal
import json


class ColoRadarProcessor:
    """ColoRadar数据集处理器"""
    
    def __init__(self, data_root: str, output_dir: str = "./processed_data"):
        """
        初始化处理器
        
        Args:
            data_root: ColoRadar数据集根目录
            output_dir: 处理后的数据保存目录
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # ColoRadar数据参数
        self.radar_params = {
            'range_bins': 256,
            'azimuth_bins': 64,
            'elevation_bins': 16,
            'max_range': 50.0,  # 50米最大距离
            'frame_rate': 10,   # 10-20Hz帧率
        }
        
    def load_sequence_info(self, sequence_name: str) -> Dict:
        """加载序列信息"""
        seq_path = self.data_root / sequence_name
        
        # 检查序列目录是否存在
        if not seq_path.exists():
            raise FileNotFoundError(f"序列 {sequence_name} 不存在于 {self.data_root}")
        
        # 加载序列信息（假设有info.json文件）
        info_file = seq_path / "info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
        else:
            # 如果没有info文件，使用默认值
            info = {
                'duration': 100,  # 秒
                'frame_count': 1000,
                'sensor_type': 'AWR1843',  # 单芯片雷达
                'resolution': 'low'
            }
        
        return info
    
    def load_radar_data(self, sequence_name: str, frame_idx: int) -> Dict:
        """
        加载单帧雷达数据
        
        Args:
            sequence_name: 序列名称
            frame_idx: 帧索引
            
        Returns:
            包含雷达数据的字典
        """
        seq_path = self.data_root / sequence_name
        
        # 尝试加载不同格式的雷达数据
        radar_data = {}
        
        # 1. 首先检查是否存在直接的数据文件
        # 支持多种可能的文件命名方式
        possible_files = [
            f"radar/frame_{frame_idx:06d}.bin",
            f"radar/frame_{frame_idx:06d}.npy", 
            f"radar/pointcloud_{frame_idx:06d}.bin",
            f"radar/heatmap_{frame_idx:06d}.npy",
            f"radar/adc_{frame_idx:06d}.bin",
            f"frame_{frame_idx:06d}.bin",  # 直接放在序列目录下
            f"frame_{frame_idx:06d}.npy"
        ]
        
        for file_pattern in possible_files:
            file_path = seq_path / file_pattern
            if file_path.exists():
                if file_pattern.endswith('.bin'):
                    radar_data['raw'] = self._load_binary_file(file_path)
                elif file_pattern.endswith('.npy'):
                    radar_data['raw'] = np.load(file_path)
                break
        
        # 2. 如果找不到标准文件，尝试加载整个目录的数据
        if not radar_data:
            radar_data = self._load_directory_data(seq_path, frame_idx)
        
        return radar_data
    
    def _load_directory_data(self, seq_path: Path, frame_idx: int) -> Dict:
        """加载目录中的所有数据文件"""
        radar_data = {}
        
        # 检查是否存在雷达数据目录
        radar_dir = seq_path / "radar"
        if radar_dir.exists():
            # 查找所有相关文件
            for file_path in radar_dir.iterdir():
                if str(frame_idx) in file_path.name:
                    if file_path.suffix == '.bin':
                        radar_data['raw'] = self._load_binary_file(file_path)
                    elif file_path.suffix == '.npy':
                        radar_data['raw'] = np.load(file_path)
        
        return radar_data
    
    def _load_binary_file(self, file_path: Path) -> np.ndarray:
        """加载二进制文件"""
        try:
            # 尝试多种数据类型
            data = np.fromfile(file_path, dtype=np.float32)
            if len(data) == 0:
                data = np.fromfile(file_path, dtype=np.complex64)
            return data
        except Exception as e:
            print(f"加载二进制文件失败: {e}")
            return np.array([])
    
    def _load_pointcloud(self, file_path: Path) -> np.ndarray:
        """加载点云数据"""
        # ColoRadar点云格式: [x, y, z, intensity, velocity]
        try:
            pointcloud = np.fromfile(file_path, dtype=np.float32)
            pointcloud = pointcloud.reshape(-1, 5)  # 每点5个值
            return pointcloud
        except Exception as e:
            print(f"加载点云数据失败: {e}")
            return np.array([])
    
    def _load_adc_data(self, file_path: Path) -> np.ndarray:
        """加载ADC原始数据"""
        try:
            # 假设ADC数据是复数格式
            adc_data = np.fromfile(file_path, dtype=np.complex64)
            return adc_data
        except Exception as e:
            print(f"加载ADC数据失败: {e}")
            return np.array([])
    
    def convert_to_complex_rd_map(self, radar_data: Dict) -> torch.Tensor:
        """
        将雷达数据转换为复数Range-Doppler Map
        
        Args:
            radar_data: 原始雷达数据字典
            
        Returns:
            complex_rd_map: (2, H, W) 复数RD Map (实部+虚部)
        """
        # 如果有ADC数据，直接处理
        if 'adc' in radar_data and len(radar_data['adc']) > 0:
            return self._process_adc_to_rd(radar_data['adc'])
        
        # 如果有热图数据，转换为复数格式
        elif 'heatmap' in radar_data and radar_data['heatmap'] is not None:
            return self._convert_heatmap_to_complex(radar_data['heatmap'])
        
        # 如果只有点云数据，转换为热图再处理
        elif 'pointcloud' in radar_data and len(radar_data['pointcloud']) > 0:
            heatmap = self._pointcloud_to_heatmap(radar_data['pointcloud'])
            return self._convert_heatmap_to_complex(heatmap)
        
        else:
            # 生成模拟数据用于测试
            return self._generate_mock_rd_map()
    
    def _process_adc_to_rd(self, adc_data: np.ndarray) -> torch.Tensor:
        """处理ADC数据生成RD Map"""
        # 假设ADC数据是复数格式，进行2D FFT
        # 这里需要根据ColoRadar的具体格式调整
        
        # 重塑为(chirps, samples)格式
        num_chirps = self.radar_params['azimuth_bins']
        num_samples = self.radar_params['range_bins']
        
        if len(adc_data) >= num_chirps * num_samples:
            adc_reshaped = adc_data[:num_chirps * num_samples].reshape(num_chirps, num_samples)
            
            # 2D FFT生成RD Map
            rd_map = np.fft.fft2(adc_reshaped)
            
            # 转换为实部+虚部
            real_part = np.real(rd_map)
            imag_part = np.imag(rd_map)
            
            # 归一化
            real_part = self._normalize_data(real_part)
            imag_part = self._normalize_data(imag_part)
            
            # 转换为PyTorch张量
            complex_rd = torch.from_numpy(np.stack([real_part, imag_part], axis=0)).float()
            return complex_rd
        
        else:
            return self._generate_mock_rd_map()
    
    def _convert_heatmap_to_complex(self, heatmap: np.ndarray) -> torch.Tensor:
        """将热图转换为复数RD Map"""
        # 假设热图是幅度信息，需要生成相位信息
        magnitude = heatmap
        
        # 生成随机相位（真实数据应该有相位信息）
        phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        
        # 转换为复数
        complex_data = magnitude * np.exp(1j * phase)
        
        # 分离实部虚部
        real_part = np.real(complex_data)
        imag_part = np.imag(complex_data)
        
        # 归一化
        real_part = self._normalize_data(real_part)
        imag_part = self._normalize_data(imag_part)
        
        # 转换为PyTorch张量
        complex_rd = torch.from_numpy(np.stack([real_part, imag_part], axis=0)).float()
        return complex_rd
    
    def _pointcloud_to_heatmap(self, pointcloud: np.ndarray) -> np.ndarray:
        """将点云转换为热图"""
        # 创建空的RD Map
        heatmap = np.zeros((self.radar_params['range_bins'], 
                           self.radar_params['azimuth_bins']))
        
        # 将点云投影到RD平面
        for point in pointcloud:
            x, y, z, intensity, velocity = point
            
            # 计算距离和角度
            range_val = np.sqrt(x**2 + y**2 + z**2)
            azimuth = np.arctan2(y, x)
            
            # 转换为bin索引
            range_bin = int((range_val / self.radar_params['max_range']) * 
                           self.radar_params['range_bins'])
            azimuth_bin = int(((azimuth + np.pi) / (2 * np.pi)) * 
                             self.radar_params['azimuth_bins'])
            
            # 确保索引在范围内
            if (0 <= range_bin < self.radar_params['range_bins'] and 
                0 <= azimuth_bin < self.radar_params['azimuth_bins']):
                heatmap[range_bin, azimuth_bin] += intensity
        
        return heatmap
    
    def _generate_mock_rd_map(self) -> torch.Tensor:
        """生成模拟RD Map用于测试"""
        H, W = self.radar_params['range_bins'], self.radar_params['azimuth_bins']
        
        # 生成随机复数数据
        real_part = np.random.randn(H, W)
        imag_part = np.random.randn(H, W)
        
        # 归一化
        real_part = self._normalize_data(real_part)
        imag_part = self._normalize_data(imag_part)
        
        return torch.from_numpy(np.stack([real_part, imag_part], axis=0)).float()
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据归一化到[-1, 1]范围"""
        if np.max(data) - np.min(data) == 0:
            return np.zeros_like(data)
        return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
    
    def generate_bev_from_lidar(self, sequence_name: str, frame_idx: int) -> Optional[torch.Tensor]:
        """
        从LiDAR数据生成BEV布局
        
        Args:
            sequence_name: 序列名称
            frame_idx: 帧索引
            
        Returns:
            bev_layout: (3, H, W) BEV布局图
        """
        seq_path = self.data_root / sequence_name
        
        # 尝试加载LiDAR点云
        lidar_file = seq_path / "lidar" / f"pointcloud_{frame_idx:06d}.bin"
        if not lidar_file.exists():
            print(f"LiDAR文件不存在: {lidar_file}")
            return None
        
        try:
            # 加载LiDAR点云
            lidar_data = np.fromfile(lidar_file, dtype=np.float32)
            lidar_data = lidar_data.reshape(-1, 4)  # [x, y, z, intensity]
            
            # 创建BEV投影
            bev_size = 256  # BEV图像尺寸
            bev_range = 50.0  # 50米范围
            
            # 创建BEV图像
            bev_image = np.zeros((3, bev_size, bev_size), dtype=np.float32)
            
            # 将点云投影到BEV
            for point in lidar_data:
                x, y, z, intensity = point
                
                # 转换为像素坐标
                u = int((x + bev_range/2) / bev_range * bev_size)
                v = int((y + bev_range/2) / bev_range * bev_size)
                
                if 0 <= u < bev_size and 0 <= v < bev_size:
                    # 根据高度和强度设置颜色
                    height_color = min(max(z / 5.0, 0), 1)  # 高度映射到颜色
                    intensity_color = min(max(intensity, 0), 1)
                    
                    bev_image[0, v, u] = height_color  # 红色通道：高度
                    bev_image[1, v, u] = intensity_color  # 绿色通道：强度
                    bev_image[2, v, u] = 1.0  # 蓝色通道：固定值
            
            return torch.from_numpy(bev_image).float()
            
        except Exception as e:
            print(f"生成BEV失败: {e}")
            return None
    
    def process_sequence(self, sequence_name: str, max_frames: int = 100) -> Dict:
        """
        处理整个序列
        
        Args:
            sequence_name: 序列名称
            max_frames: 最大处理帧数
            
        Returns:
            处理结果字典
        """
        print(f"开始处理序列: {sequence_name}")
        
        sequence_info = self.load_sequence_info(sequence_name)
        total_frames = min(sequence_info.get('frame_count', max_frames), max_frames)
        
        processed_data = {
            'radar_maps': [],
            'bev_layouts': [],
            'frame_indices': []
        }
        
        for frame_idx in range(total_frames):
            try:
                # 处理雷达数据
                radar_data = self.load_radar_data(sequence_name, frame_idx)
                complex_rd_map = self.convert_to_complex_rd_map(radar_data)
                
                # 生成BEV布局
                bev_layout = self.generate_bev_from_lidar(sequence_name, frame_idx)
                if bev_layout is None:
                    # 如果没有LiDAR数据，生成模拟BEV
                    bev_layout = torch.randn(3, 256, 256)
                
                processed_data['radar_maps'].append(complex_rd_map)
                processed_data['bev_layouts'].append(bev_layout)
                processed_data['frame_indices'].append(frame_idx)
                
                if frame_idx % 10 == 0:
                    print(f"已处理 {frame_idx}/{total_frames} 帧")
                    
            except Exception as e:
                print(f"处理帧 {frame_idx} 失败: {e}")
                continue
        
        # 保存处理结果
        output_file = self.output_dir / f"{sequence_name}_processed.pt"
        torch.save(processed_data, output_file)
        
        print(f"序列处理完成，保存到: {output_file}")
        return processed_data
    
    def visualize_sample(self, radar_map: torch.Tensor, bev_layout: torch.Tensor, 
                        save_path: Optional[str] = None):
        """可视化样本数据"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 雷达数据可视化
        radar_real = radar_map[0].numpy()
        radar_imag = radar_map[1].numpy()
        radar_magnitude = np.sqrt(radar_real**2 + radar_imag**2)
        radar_phase = np.arctan2(radar_imag, radar_real)
        
        axes[0, 0].imshow(radar_real, cmap='viridis')
        axes[0, 0].set_title('雷达实部')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(radar_imag, cmap='viridis')
        axes[0, 1].set_title('雷达虚部')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(radar_magnitude, cmap='hot')
        axes[1, 0].set_title('雷达幅度')
        axes[1, 0].axis('off')
        
        # BEV可视化
        bev_rgb = bev_layout.permute(1, 2, 0).numpy()
        axes[1, 1].imshow(bev_rgb)
        axes[1, 1].set_title('BEV布局')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果保存到: {save_path}")
        
        plt.show()


def main():
    """主函数 - 测试数据处理器"""
    # 示例用法
    processor = ColoRadarProcessor(
        data_root="D:/ColoRadar/data",  # 修改为实际路径
        output_dir="./processed_data"
    )
    
    # 处理测试序列
    test_sequence = "ec_hallways_run0"
    processed_data = processor.process_sequence(test_sequence, max_frames=50)
    
    # 可视化第一个样本
    if len(processed_data['radar_maps']) > 0:
        processor.visualize_sample(
            processed_data['radar_maps'][0],
            processed_data['bev_layouts'][0],
            save_path=f"./visualization_{test_sequence}.png"
        )


if __name__ == "__main__":
    main()