"""
ADC数据读取模块
专门用于读取和处理CoLoRadar数据集的FMCW原始ADC数据
"""

import numpy as np
import torch
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import struct
import warnings

# 真实物理地址配置
REAL_PATHS = {
    # CoLoRadar数据集路径
    "COLO_RADAR_DATASET_PATH": r"D:\datasets\CoLoRadar",
    "COLO_RADAR_SEQUENCES": [
        "2_28_2021_outdoors_run9",
        "2_28_2021_outdoors_run10", 
        "2_28_2021_outdoors_run11",
        "2_28_2021_outdoors_run12"
    ],
    
    # 项目相关路径
    "PROJECT_ROOT": r"D:\桌面\PhysRadar-DiT-master",
    "RADAR_PROCESSING_DIR": r"D:\桌面\PhysRadar-DiT-master\radar_processing",
    "OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output",
    "MODEL_DIR": r"D:\桌面\PhysRadar-DiT-master\models",
    
    # 预训练模型路径
    "PRETRAINED_MODELS": {
        "rd_processor": r"D:\桌面\PhysRadar-DiT-master\models\rd_processor.pth",
        "ra_processor": r"D:\桌面\PhysRadar-DiT-master\models\ra_processor.pth",
        "adc_reader": r"D:\桌面\PhysRadar-DiT-master\models\adc_reader.pth"
    }
}


class ADCReader:
    """
    ADC数据读取器类
    支持多种格式的FMCW雷达原始ADC数据读取和处理
    """
    
    def __init__(self, 
                 sample_rate: float = 10e6,
                 chirp_duration: float = 50e-6,
                 bandwidth: float = 1e9,
                 center_freq: float = 77e9,
                 num_tx: int = 3,
                 num_rx: int = 4,
                 num_samples_per_chirp: int = 256,
                 num_chirps_per_frame: int = 128,
                 dataset_path: str = REAL_PATHS["COLO_RADAR_DATASET_PATH"]):
        """
        初始化ADC读取器
        
        Args:
            sample_rate: 采样率 (Hz)
            chirp_duration: 单个chirp持续时间 (s)
            bandwidth: 带宽 (Hz)
            center_freq: 中心频率 (Hz)
            num_tx: 发射天线数量
            num_rx: 接收天线数量
            num_samples_per_chirp: 每个chirp的采样点数
            num_chirps_per_frame: 每帧的chirp数量
            dataset_path: 真实数据集路径
        """
        self.sample_rate = sample_rate
        self.chirp_duration = chirp_duration
        self.bandwidth = bandwidth
        self.center_freq = center_freq
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_samples_per_chirp = num_samples_per_chirp
        self.num_chirps_per_frame = num_chirps_per_frame
        self.dataset_path = dataset_path
        self.real_paths = REAL_PATHS
        
        # 计算雷达参数
        self.wavelength = 3e8 / center_freq  # 波长
        self.range_resolution = 3e8 / (2 * bandwidth)  # 距离分辨率
        self.max_range = sample_rate * 3e8 / (2 * bandwidth)  # 最大探测距离
        self.max_velocity = self.wavelength / (4 * chirp_duration)  # 最大速度
        
        print(f"ADC Reader初始化完成:")
        print(f"  采样率: {sample_rate/1e6:.1f} MHz")
        print(f"  带宽: {bandwidth/1e9:.1f} GHz")
        print(f"  中心频率: {center_freq/1e9:.1f} GHz")
        print(f"  距离分辨率: {self.range_resolution:.3f} m")
        print(f"  最大探测距离: {self.max_range:.1f} m")
        print(f"  最大速度: {self.max_velocity:.1f} m/s")
        print(f"  数据集路径: {self.dataset_path}")
    
    def read_binary_file(self, file_path: str, 
                        data_format: str = "complex64",
                        endian: str = "little") -> np.ndarray:
        """
        读取二进制格式的ADC数据文件
        
        Args:
            file_path: 文件路径
            data_format: 数据格式 ("complex64", "int16", "float32")
            endian: 字节序 ("little", "big")
            
        Returns:
            adc_data: 形状为 (num_frames, num_tx, num_rx, num_chirps, num_samples) 的ADC数据
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ADC数据文件不存在: {file_path}")
        
        print(f"读取ADC数据文件: {file_path}")
        
        # 根据数据格式确定数据类型和字节大小
        dtype_map = {
            "complex64": np.complex64,
            "int16": np.int16,
            "float32": np.float32
        }
        
        if data_format not in dtype_map:
            raise ValueError(f"不支持的数据格式: {data_format}")
        
        dtype = dtype_map[data_format]
        bytes_per_sample = np.dtype(dtype).itemsize
        
        # 计算文件大小和预期数据量
        file_size = os.path.getsize(file_path)
        total_samples = file_size // bytes_per_sample
        
        # 计算帧数
        samples_per_frame = self.num_tx * self.num_rx * self.num_chirps_per_frame * self.num_samples_per_chirp
        num_frames = total_samples // samples_per_frame
        
        if num_frames == 0:
            raise ValueError("文件大小与预期数据格式不匹配")
        
        print(f"  文件大小: {file_size/1024/1024:.2f} MB")
        print(f"  总采样点数: {total_samples}")
        print(f"  每帧采样点数: {samples_per_frame}")
        print(f"  帧数: {num_frames}")
        
        # 读取数据
        with open(file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=dtype)
        
        # 重塑数据形状
        adc_data = raw_data.reshape(num_frames, self.num_tx, self.num_rx, 
                                   self.num_chirps_per_frame, self.num_samples_per_chirp)
        
        print(f"  ADC数据形状: {adc_data.shape}")
        return adc_data
    
    def read_colo_radar_dataset(self, 
                               sequence_name: str = "2_28_2021_outdoors_run9",
                               dataset_path: str = None) -> Dict:
        """
        读取CoLoRadar数据集的ADC数据
        
        Args:
            sequence_name: 序列名称
            dataset_path: 数据集路径，如果为None则使用默认路径
            
        Returns:
            radar_data: 包含ADC数据和元数据的字典
        """
        if dataset_path is None:
            dataset_path = self.dataset_path
            
        sequence_path = os.path.join(dataset_path, sequence_name)
        if not os.path.exists(sequence_path):
            # 检查序列是否在预定义列表中
            if sequence_name not in self.real_paths["COLO_RADAR_SEQUENCES"]:
                warnings.warn(f"序列 {sequence_name} 不在预定义列表中")
            
            raise FileNotFoundError(f"序列路径不存在: {sequence_path}")
        
        print(f"读取CoLoRadar序列: {sequence_name}")
        print(f"  完整路径: {sequence_path}")
        
        # 查找ADC数据文件
        adc_files = []
        for file in os.listdir(sequence_path):
            if file.endswith('.bin') or file.endswith('.dat') or 'adc' in file.lower():
                full_path = os.path.join(sequence_path, file)
                adc_files.append(full_path)
                print(f"  找到ADC文件: {file}")
        
        if not adc_files:
            raise FileNotFoundError(f"在序列 {sequence_name} 中未找到ADC数据文件")
        
        # 读取第一个ADC文件
        adc_data = self.read_binary_file(adc_files[0])
        
        # 读取校准和元数据
        calibration_data = self._read_calibration_data(sequence_path)
        metadata = self._read_metadata(sequence_path)
        
        radar_data = {
            'adc_data': adc_data,
            'calibration': calibration_data,
            'metadata': metadata,
            'file_path': adc_files[0],
            'sequence_name': sequence_name,
            'sequence_path': sequence_path,
            'dataset_path': dataset_path
        }
        
        return radar_data
    
    def _read_calibration_data(self, sequence_path: str) -> Dict:
        """读取校准数据"""
        calibration_file = os.path.join(sequence_path, 'calibration.json')
        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                return json.load(f)
        else:
            warnings.warn(f"未找到校准文件: {calibration_file}")
            return {}
    
    def _read_metadata(self, sequence_path: str) -> Dict:
        """读取元数据"""
        metadata_file = os.path.join(sequence_path, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            # 生成默认元数据
            return {
                'sample_rate': self.sample_rate,
                'chirp_duration': self.chirp_duration,
                'bandwidth': self.bandwidth,
                'center_freq': self.center_freq,
                'num_tx': self.num_tx,
                'num_rx': self.num_rx,
                'num_samples_per_chirp': self.num_samples_per_chirp,
                'num_chirps_per_frame': self.num_chirps_per_frame
            }
    
    def apply_calibration(self, adc_data: np.ndarray, calibration_data: Dict) -> np.ndarray:
        """
        应用校准数据到ADC数据
        
        Args:
            adc_data: 原始ADC数据
            calibration_data: 校准数据字典
            
        Returns:
            calibrated_data: 校准后的ADC数据
        """
        print("应用校准数据...")
        
        calibrated_data = adc_data.copy()
        
        # 应用相位校准
        if 'phase_calibration' in calibration_data:
            phase_cal = np.array(calibration_data['phase_calibration'])
            if phase_cal.shape == adc_data.shape[1:4]:  # (tx, rx, chirps)
                calibrated_data = calibrated_data * np.exp(1j * phase_cal[np.newaxis, :, :, :, np.newaxis])
        
        # 应用幅度校准
        if 'amplitude_calibration' in calibration_data:
            amp_cal = np.array(calibration_data['amplitude_calibration'])
            if amp_cal.shape == adc_data.shape[1:4]:  # (tx, rx, chirps)
                calibrated_data = calibrated_data * amp_cal[np.newaxis, :, :, :, np.newaxis]
        
        # 应用DC偏移校准
        if 'dc_offset' in calibration_data:
            dc_offset = np.array(calibration_data['dc_offset'])
            if dc_offset.shape == adc_data.shape[1:4]:  # (tx, rx, chirps)
                calibrated_data = calibrated_data - dc_offset[np.newaxis, :, :, :, np.newaxis]
        
        print("✓ 校准完成")
        return calibrated_data
    
    def preprocess_adc_data(self, adc_data: np.ndarray, 
                           apply_window: bool = True,
                           remove_dc: bool = True) -> np.ndarray:
        """
        预处理ADC数据
        
        Args:
            adc_data: 原始ADC数据
            apply_window: 是否应用窗函数
            remove_dc: 是否移除DC分量
            
        Returns:
            processed_data: 预处理后的ADC数据
        """
        print("预处理ADC数据...")
        
        processed_data = adc_data.copy()
        
        # 移除DC分量
        if remove_dc:
            dc_offset = np.mean(processed_data, axis=-1, keepdims=True)
            processed_data = processed_data - dc_offset
            print("  ✓ 移除DC分量")
        
        # 应用窗函数
        if apply_window:
            num_samples = adc_data.shape[-1]  # 获取样本数
            window = np.hanning(num_samples)
            # 确保窗口形状与数据匹配
            window_shape = (1,) * (len(adc_data.shape) - 1) + (num_samples,)
            processed_data = processed_data * window.reshape(window_shape)
            print("  ✓ 应用Hanning窗")
        
        # 数据归一化
        max_val = np.max(np.abs(processed_data))
        if max_val > 0:
            processed_data = processed_data / max_val
            print("  ✓ 数据归一化")
        
        print("✓ 预处理完成")
        return processed_data
    
    def convert_to_complex_tensor(self, adc_data: np.ndarray) -> torch.Tensor:
        """
        将ADC数据转换为PyTorch复数张量
        
        Args:
            adc_data: ADC数据数组，形状为 (frames, tx, rx, chirps, samples)
            
        Returns:
            complex_tensor: 形状为 (B, 2, T, H, W) 的复数张量
        """
        # 检查输入数据形状
        if len(adc_data.shape) != 5:
            raise ValueError(f"ADC数据形状必须为5D (frames, tx, rx, chirps, samples)，当前形状: {adc_data.shape}")
        
        num_frames, num_tx, num_rx, num_chirps, num_samples = adc_data.shape
        num_antennas = num_tx * num_rx
        
        # 重塑为 (frames, chirps, samples, antennas)
        reshaped_data = adc_data.transpose(0, 3, 4, 1, 2).reshape(
            num_frames, num_chirps, num_samples, num_antennas)
        
        # 分离实部和虚部
        real_part = np.real(reshaped_data)
        imag_part = np.imag(reshaped_data)
        
        # 合并为 (frames, 2, chirps, samples, antennas)
        complex_data = np.stack([real_part, imag_part], axis=1)
        
        # 转换为PyTorch张量
        complex_tensor = torch.from_numpy(complex_data).float()
        
        print(f"转换为复数张量: {adc_data.shape} -> {complex_tensor.shape}")
        return complex_tensor

    def convert_to_tensor(self, adc_data: np.ndarray) -> torch.Tensor:
        """
        将ADC数据转换为PyTorch张量（convert_to_complex_tensor的别名）
        
        Args:
            adc_data: ADC数据数组
            
        Returns:
            tensor: 形状为 (B, 2, T, H, W) 的张量
        """
        return self.convert_to_complex_tensor(adc_data)


def test_adc_reader():
    """测试ADC读取器功能"""
    print("测试ADC读取器...")
    
    # 创建ADC读取器实例
    reader = ADCReader()
    
    # 测试参数计算
    print(f"波长: {reader.wavelength:.4f} m")
    print(f"距离分辨率: {reader.range_resolution:.3f} m")
    print(f"最大速度: {reader.max_velocity:.1f} m/s")
    
    # 测试数据生成（模拟）
    print("\n生成模拟ADC数据...")
    num_frames = 2
    simulated_data = np.random.randn(num_frames, reader.num_tx, reader.num_rx, 
                                   reader.num_chirps_per_frame, reader.num_samples_per_chirp) + \
                    1j * np.random.randn(num_frames, reader.num_tx, reader.num_rx, 
                                       reader.num_chirps_per_frame, reader.num_samples_per_chirp)
    
    print(f"模拟数据形状: {simulated_data.shape}")
    
    # 测试预处理
    processed_data = reader.preprocess_adc_data(simulated_data)
    print(f"预处理后数据形状: {processed_data.shape}")
    
    # 测试转换为张量
    tensor_data = reader.convert_to_complex_tensor(processed_data)
    print(f"张量数据形状: {tensor_data.shape}")
    
    print("✓ ADC读取器测试完成")


if __name__ == "__main__":
    test_adc_reader()