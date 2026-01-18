"""
RD（距离-多普勒）数据处理模块
用于生成距离-多普勒矩阵和相关的雷达信号处理
"""

import numpy as np
import torch
import torch.fft as fft
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy import signal
import scipy.ndimage as ndimage
import warnings
import os
from datetime import datetime

# 真实物理地址配置
REAL_PATHS = {
    # 项目相关路径
    "PROJECT_ROOT": r"D:\桌面\PhysRadar-DiT-master",
    "RADAR_PROCESSING_DIR": r"D:\桌面\PhysRadar-DiT-master\radar_processing",
    "OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output",
    "RD_OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output\rd_results",
    "MODEL_DIR": r"D:\桌面\PhysRadar-DiT-master\models",
    
    # 预训练模型路径
    "PRETRAINED_MODELS": {
        "rd_processor": r"D:\桌面\PhysRadar-DiT-master\models\rd_processor.pth",
        "ra_processor": r"D:\桌面\PhysRadar-DiT-master\models\ra_processor.pth"
    }
}


class RDProcessor:
    """
    RD数据处理类
    实现距离-多普勒处理和相关算法
    """
    
    def __init__(self, 
                 sample_rate: float = 10e6,
                 chirp_duration: float = 50e-6,
                 bandwidth: float = 1e9,
                 center_freq: float = 77e9,
                 num_chirps: int = 128,
                 num_samples: int = 256,
                 output_dir: str = REAL_PATHS["RD_OUTPUT_DIR"]):
        """
        初始化RD处理器
        
        Args:
            sample_rate: 采样率 (Hz)
            chirp_duration: chirp持续时间 (s)
            bandwidth: 带宽 (Hz)
            center_freq: 中心频率 (Hz)
            num_chirps: chirp数量
            num_samples: 每个chirp的采样点数
            output_dir: 真实输出目录路径
        """
        self.sample_rate = sample_rate
        self.chirp_duration = chirp_duration
        self.bandwidth = bandwidth
        self.center_freq = center_freq
        self.num_chirps = num_chirps
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.real_paths = REAL_PATHS
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 计算雷达参数
        self.wavelength = 3e8 / center_freq
        self.range_resolution = 3e8 / (2 * bandwidth)
        self.max_range = sample_rate * 3e8 / (2 * bandwidth)
        self.velocity_resolution = self.wavelength / (2 * num_chirps * chirp_duration)
        self.max_velocity = self.wavelength / (4 * chirp_duration)
        
        # 计算频率轴
        self.range_axis = np.fft.fftshift(np.fft.fftfreq(num_samples, 1/sample_rate)) * 3e8 / (2 * bandwidth)
        self.doppler_axis = np.fft.fftshift(np.fft.fftfreq(num_chirps, chirp_duration)) * self.wavelength / 2
        
        print(f"RD Processor初始化完成:")
        print(f"  距离分辨率: {self.range_resolution:.3f} m")
        print(f"  速度分辨率: {self.velocity_resolution:.3f} m/s")
        print(f"  最大探测距离: {self.max_range:.1f} m")
        print(f"  最大速度: {self.max_velocity:.1f} m/s")
        print(f"  输出目录: {self.output_dir}")
    
    def range_fft(self, adc_data: np.ndarray,
                  apply_window: bool = True,
                  zero_padding: int = 0) -> np.ndarray:
        """
        距离维FFT处理
        
        Args:
            adc_data: ADC数据，形状为 (num_chirps, num_samples)
            apply_window: 是否应用窗函数
            zero_padding: 零填充点数
            
        Returns:
            range_profile: 距离剖面，形状为 (num_chirps, fft_points)
        """
        print("执行距离维FFT...")
        
        original_shape = adc_data.shape
        num_samples = adc_data.shape[-1]  # 获取输入数据的样本数
        fft_points = num_samples + zero_padding
        
        # 应用窗函数 - 沿samples维度（axis=-1）
        if apply_window:
            window = np.hanning(num_samples)  # 根据实际样本数生成窗口
            window = window.reshape(1, -1)    # 形状变为 (1, samples)
            windowed_data = adc_data * window
        else:
            windowed_data = adc_data
        
        # 零填充 - 沿samples维度
        if zero_padding > 0:
            padded_data = np.zeros((adc_data.shape[0], fft_points), dtype=adc_data.dtype)
            padded_data[:, :num_samples] = windowed_data
        else:
            padded_data = windowed_data
            fft_points = num_samples
        
        # 执行FFT - 沿axis=-1（samples维度）
        range_fft = np.fft.fft(padded_data, axis=-1)
        
        # fftshift - 仅在range轴（axis=-1）
        range_fft = np.fft.fftshift(range_fft, axes=-1)
        
        print(f"  输入形状: {original_shape}")
        print(f"  输出形状: {range_fft.shape}")
        print(f"  FFT点数: {fft_points}")
        print(f"  窗函数形状: {window.shape if apply_window else '无窗函数'}")
        
        return range_fft

    def doppler_fft(self, range_data: np.ndarray,
                   apply_window: bool = True,
                   zero_padding: int = 0) -> np.ndarray:
        """
        多普勒维FFT处理
        
        Args:
            range_data: 距离剖面数据，形状必须为 (num_chirps, range_bins)
            apply_window: 是否应用窗函数
            zero_padding: 零填充点数
            
        Returns:
            rd_matrix: 距离-多普勒矩阵，形状为 (doppler_bins, range_bins)
        """
        print("执行多普勒维FFT...")
        
        # 检查输入数据形状
        if len(range_data.shape) != 2:
            raise ValueError(f"输入数据形状必须为 (chirps, range_bins)，当前形状: {range_data.shape}")
        
        original_shape = range_data.shape
        num_chirps = range_data.shape[0]  # 获取实际的chirps数
        doppler_bins = num_chirps + zero_padding
        
        # 应用窗函数 - 沿chirps维度（axis=0）
        if apply_window:
            window = np.hanning(num_chirps)  # 窗函数长度等于chirps数
            window = window.reshape(-1, 1)   # 形状变为 (chirps, 1)
            windowed_data = range_data * window
        else:
            windowed_data = range_data
        
        # 零填充 - 沿chirps维度
        if zero_padding > 0:
            padded_data = np.zeros((doppler_bins, range_data.shape[1]), dtype=range_data.dtype)
            padded_data[:num_chirps, :] = windowed_data
        else:
            padded_data = windowed_data
            doppler_bins = num_chirps
        
        # 执行FFT - 沿axis=0（chirps维度）
        doppler_fft = np.fft.fft(padded_data, axis=0)
        
        # fftshift - 仅在doppler轴（axis=0）
        doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
        
        print(f"  输入形状: {original_shape}")
        print(f"  输出形状: {doppler_fft.shape}")
        print(f"  多普勒点数: {doppler_bins}")
        print(f"  窗函数形状: {window.shape if apply_window else '无窗函数'}")
        
        return doppler_fft

    def generate_rd_matrix(self, adc_data: np.ndarray,
                          range_fft_config: Dict = None,
                          doppler_fft_config: Dict = None) -> np.ndarray:
        """
        生成完整的距离-多普勒矩阵
        
        Args:
            adc_data: ADC数据，形状必须为 (chirps, samples)
            range_fft_config: 距离FFT配置
            doppler_fft_config: 多普勒FFT配置
            
        Returns:
            rd_matrix: 距离-多普勒矩阵，形状为 (D, R)
        """
        if range_fft_config is None:
            range_fft_config = {'apply_window': True, 'zero_padding': 0}
        if doppler_fft_config is None:
            doppler_fft_config = {'apply_window': True, 'zero_padding': 0}
        
        print("生成距离-多普勒矩阵...")
        
        # 检查输入数据形状
        if len(adc_data.shape) != 2:
            raise ValueError(f"ADC数据形状必须为 (chirps, samples)，当前形状: {adc_data.shape}")
        
        # 距离FFT
        range_profile = self.range_fft(adc_data, **range_fft_config)
        
        # 多普勒FFT
        rd_matrix = self.doppler_fft(range_profile, **doppler_fft_config)
        
        # 统一对结果取幅度并做对数压缩
        rd_magnitude = np.abs(rd_matrix)
        rd_log_compressed = 20 * np.log10(rd_magnitude + 1e-10)  # 避免log(0)
        
        print(f"  RD矩阵形状: {rd_log_compressed.shape}")
        print(f"  Doppler bins: {rd_log_compressed.shape[0]}")
        print(f"  Range bins: {rd_log_compressed.shape[1]}")
        
        return rd_log_compressed


    def cfar_detection(self, rd_matrix: np.ndarray,
                      guard_cells: int = 2,
                      training_cells: int = 10,
                      false_alarm_rate: float = 1e-6) -> np.ndarray:
        """
        CFAR（恒虚警率）检测
        
        Args:
            rd_matrix: 距离-多普勒矩阵
            guard_cells: 保护单元数
            training_cells: 训练单元数
            false_alarm_rate: 虚警率
            
        Returns:
            detection_mask: 检测掩码
        """
        print("执行CFAR检测...")
        
        # 计算CFAR阈值
        threshold_factor = -np.log(false_alarm_rate)
        
        # 获取幅度
        magnitude = np.abs(rd_matrix)
        detection_mask = np.zeros_like(magnitude, dtype=bool)
        
        # 二维CFAR处理
        for i in range(guard_cells + training_cells, magnitude.shape[0] - guard_cells - training_cells):
            for j in range(guard_cells + training_cells, magnitude.shape[1] - guard_cells - training_cells):
                # 获取训练单元
                training_region = np.concatenate([
                    magnitude[i-training_cells-guard_cells:i-guard_cells, 
                             j-training_cells-guard_cells:j+training_cells+guard_cells+1].flatten(),
                    magnitude[i+guard_cells+1:i+training_cells+guard_cells+1, 
                             j-training_cells-guard_cells:j+training_cells+guard_cells+1].flatten(),
                    magnitude[i-guard_cells:i+guard_cells+1, 
                             j-training_cells-guard_cells:j-guard_cells].flatten(),
                    magnitude[i-guard_cells:i+guard_cells+1, 
                             j+guard_cells+1:j+training_cells+guard_cells+1].flatten()
                ])
                
                # 计算噪声水平
                noise_level = np.mean(training_region)
                threshold = noise_level * threshold_factor
                
                # 检测
                if magnitude[i, j] > threshold:
                    detection_mask[i, j] = True
        
        print(f"  检测到目标数: {np.sum(detection_mask)}")
        return detection_mask
    
    def peak_detection(self, rd_matrix: np.ndarray,
                      min_magnitude: float = 0.1,
                      min_distance: int = 3) -> List[Tuple]:
        """
        峰值检测
        
        Args:
            rd_matrix: 距离-多普勒矩阵
            min_magnitude: 最小幅度阈值
            min_distance: 最小距离（像素）
            
        Returns:
            peaks: 峰值列表 [(range_idx, doppler_idx, magnitude), ...]
        """
        magnitude = np.abs(rd_matrix)
        
        # 使用scipy的argrelextrema函数进行峰值检测（兼容性更好）
        try:
            from scipy.signal import argrelextrema
            
            # 在行方向（距离轴）找局部最大值
            range_peaks = argrelextrema(magnitude, np.greater, axis=0, order=min_distance)
            # 在列方向（多普勒轴）找局部最大值
            doppler_peaks = argrelextrema(magnitude, np.greater, axis=1, order=min_distance)
            
            # 取两个方向的交集
            peak_mask = np.zeros_like(magnitude, dtype=bool)
            peak_mask[range_peaks] = True
            peak_mask &= np.zeros_like(magnitude, dtype=bool)
            peak_mask[:, doppler_peaks[1]] = True
            
            # 获取峰值坐标
            peaks = np.where(peak_mask)
            
        except ImportError:
            # 如果scipy.signal.argrelextrema不可用，使用简单的局部最大值检测
            peaks = self._simple_peak_detection(magnitude, min_distance, min_magnitude)
        
        peak_list = []
        for i in range(len(peaks[0])):
            range_idx, doppler_idx = peaks[0][i], peaks[1][i]
            mag = magnitude[range_idx, doppler_idx]
            
            # 应用幅度阈值
            if mag >= min_magnitude * np.max(magnitude):
                peak_list.append((range_idx, doppler_idx, mag))
        
        print(f"检测到 {len(peak_list)} 个峰值")
        return peak_list
    
    def _simple_peak_detection(self, magnitude: np.ndarray, 
                              min_distance: int, 
                              min_magnitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        简单的局部峰值检测实现
        
        Args:
            magnitude: 幅度矩阵
            min_distance: 最小距离
            min_magnitude: 最小幅度阈值
            
        Returns:
            peaks: 峰值坐标 (range_indices, doppler_indices)
        """
        height_threshold = min_magnitude * np.max(magnitude)
        peaks_range = []
        peaks_doppler = []
        
        # 遍历矩阵，寻找局部最大值
        for i in range(min_distance, magnitude.shape[0] - min_distance):
            for j in range(min_distance, magnitude.shape[1] - min_distance):
                current_val = magnitude[i, j]
                
                # 检查是否大于阈值
                if current_val < height_threshold:
                    continue
                
                # 检查是否是局部最大值
                is_peak = True
                for di in range(-min_distance, min_distance + 1):
                    for dj in range(-min_distance, min_distance + 1):
                        if di == 0 and dj == 0:
                            continue
                        if magnitude[i + di, j + dj] >= current_val:
                            is_peak = False
                            break
                    if not is_peak:
                        break
                
                if is_peak:
                    peaks_range.append(i)
                    peaks_doppler.append(j)
        
        return np.array(peaks_range), np.array(peaks_doppler)

    def convert_to_physical_coordinates(self, range_idx: int, doppler_idx: int) -> Tuple[float, float]:
        """
        将索引坐标转换为物理坐标
        
        Args:
            range_idx: 距离索引
            doppler_idx: 多普勒索引
            
        Returns:
            (range, velocity): 物理距离和速度
        """
        range_val = self.range_axis[range_idx]
        velocity_val = self.doppler_axis[doppler_idx]
        
        return range_val, velocity_val
    
    def generate_rd_video(self, adc_data: np.ndarray,
                         frame_stride: int = 1) -> np.ndarray:
        """
        生成RD视频序列
        
        Args:
            adc_data: ADC数据，形状为 (num_frames, ...)
            frame_stride: 帧步长
            
        Returns:
            rd_video: RD视频，形状为 (num_frames, doppler_bins, range_bins)
        """
        num_frames = adc_data.shape[0]
        rd_frames = []
        
        print(f"生成RD视频序列，共 {num_frames} 帧...")
        
        for i in range(0, num_frames, frame_stride):
            frame_data = adc_data[i]
            rd_matrix = self.generate_rd_matrix(frame_data)
            rd_magnitude = np.abs(rd_matrix)
            rd_frames.append(rd_magnitude)
        
        rd_video = np.stack(rd_frames, axis=0)
        print(f"  RD视频形状: {rd_video.shape}")
        return rd_video
    
    def plot_rd_matrix(self, rd_matrix: np.ndarray,
                      title: str = "距离-多普勒矩阵",
                      save_path: Optional[str] = None,
                      sequence_name: str = "test_sequence"):
        """
        绘制距离-多普勒矩阵
        
        Args:
            rd_matrix: RD矩阵
            title: 图像标题
            save_path: 保存路径（可选）
            sequence_name: 序列名称，用于自动生成保存路径
        """
        magnitude = 20 * np.log10(np.abs(rd_matrix) + 1e-10)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(magnitude, 
                  extent=[self.range_axis[0], self.range_axis[-1], 
                         self.doppler_axis[0], self.doppler_axis[-1]],
                  aspect='auto', cmap='jet', origin='lower')
        plt.colorbar(label='幅度 (dB)')
        plt.xlabel('距离 (m)')
        plt.ylabel('速度 (m/s)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # 自动生成保存路径
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"rd_matrix_{sequence_name}_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存RD图: {save_path}")
        
        plt.show()
        
        return save_path


def test_rd_processor():
    """测试RD处理器功能"""
    print("测试RD处理器...")
    
    # 创建RD处理器实例
    processor = RDProcessor()
    
    # 生成模拟ADC数据
    num_frames = 1
    num_tx = 3
    num_rx = 4
    num_chirps = processor.num_chirps
    num_samples = processor.num_samples
    
    # 创建包含目标信号的模拟数据
    adc_data = np.random.randn(num_frames, num_tx, num_rx, num_chirps, num_samples) + \
               1j * np.random.randn(num_frames, num_tx, num_rx, num_chirps, num_samples)
    
    # 添加目标信号（在特定距离和速度）
    target_range = 50  # 米
    target_velocity = 10  # 米/秒
    
    range_idx = np.argmin(np.abs(processor.range_axis - target_range))
    doppler_idx = np.argmin(np.abs(processor.doppler_axis - target_velocity))
    
    # 在特定天线和chirp上添加目标信号
    adc_data[0, 0, 0, doppler_idx, range_idx] += 10 + 10j
    
    print(f"模拟ADC数据形状: {adc_data.shape}")
    
    # 测试单个天线的RD处理
    single_antenna_data = adc_data[0, 0, 0]  # (chirps, samples)
    rd_matrix = processor.generate_rd_matrix(single_antenna_data)
    
    print(f"RD矩阵形状: {rd_matrix.shape}")
    
    # 测试峰值检测
    peaks = processor.peak_detection(rd_matrix)
    print(f"检测到 {len(peaks)} 个峰值")
    
    # 测试CFAR检测
    detection_mask = processor.cfar_detection(rd_matrix)
    print(f"CFAR检测到 {np.sum(detection_mask)} 个目标")
    
    # 绘制RD矩阵
    processor.plot_rd_matrix(rd_matrix, "测试RD矩阵")
    
    print("✓ RD处理器测试完成")


if __name__ == "__main__":
    test_rd_processor()