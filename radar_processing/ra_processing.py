"""
RA数据处理模块  
所属流程：ADC → Range FFT → Angle FFT → RA谱
物理对应：FMCW雷达的距离-角度处理，用于检测目标的距离和方位角信息
用于：真实雷达监督信号的距离-角度谱生成
"""

import numpy as np


def adc_to_ra(adc: np.ndarray) -> np.ndarray:
    """
    将ADC数据转换为距离-角度(RA)热图
    
    Args:
        adc: 输入ADC数据，形状为[n_rx, n_chirp, n_sample]的复数数组
        
    Returns:
        np.ndarray: 形状为[N_range, N_angle]的RA热图，log尺度
        
    Processing Steps:
        1. Range FFT - 距离信息提取
        2. Angle FFT (RX维度) - 角度信息提取，假设ULA阵列
        3. 能量积累和log尺度转换
    """
    n_rx, n_chirp, n_sample = adc.shape
    
    # 步骤1: Range FFT (快时间维度)
    # 物理意义：提取目标的距离信息
    range_fft = np.fft.fft(adc, axis=2)
    
    # 步骤2: Angle FFT (RX维度)，假设ULA均匀线阵
    # 物理意义：利用天线阵列的相位差提取角度信息
    angle_fft = np.fft.fft(range_fft, axis=0)
    
    # fftshift将零频分量移到中心
    ra_spectrum = np.fft.fftshift(angle_fft, axes=0)
    ra_spectrum = np.fft.fftshift(ra_spectrum, axes=2)
    
    # 多chirp非相干积累
    ra_energy = np.sum(np.abs(ra_spectrum), axis=1)
    
    # log尺度转换，便于可视化
    epsilon = 1e-12
    ra_log = 20 * np.log10(np.abs(ra_energy) + epsilon)
    
    return ra_log.astype(np.float32)


"""
RA处理模块 - 距离-角度矩阵处理
实现Range-Angle矩阵的生成、处理和可视化
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import cv2
import os
from datetime import datetime

# 真实物理地址配置
REAL_PATHS = {
    # 项目相关路径
    "PROJECT_ROOT": r"D:\桌面\PhysRadar-DiT-master",
    "RADAR_PROCESSING_DIR": r"D:\桌面\PhysRadar-DiT-master\radar_processing",
    "OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output",
    "RA_OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output\ra_results",
    "VIDEO_OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output\videos",
    "MODEL_DIR": r"D:\桌面\PhysRadar-DiT-master\models",
    
    # 预训练模型路径
    "PRETRAINED_MODELS": {
        "ra_processor": r"D:\桌面\PhysRadar-DiT-master\models\ra_processor.pth",
        "rd_processor": r"D:\桌面\PhysRadar-DiT-master\models\rd_processor.pth"
    },
    
    # 天线配置路径
    "ANTENNA_CONFIG": r"D:\桌面\PhysRadar-DiT-master\config\antenna_positions.npy"
}


class RAProcessor:
    """
    RA处理器 - 生成和处理距离-角度矩阵
    """
    
    def __init__(self, 
                 range_bins: int = 256,
                 angle_bins: int = 180,
                 range_resolution: float = 0.1,
                 angle_resolution: float = 1.0,
                 fft_size: int = 1024,
                 output_dir: str = REAL_PATHS["RA_OUTPUT_DIR"],
                 video_output_dir: str = REAL_PATHS["VIDEO_OUTPUT_DIR"],
                 antenna_config_path: str = REAL_PATHS["ANTENNA_CONFIG"]):
        """
        初始化RA处理器
        
        Args:
            range_bins: 距离维度大小
            angle_bins: 角度维度大小
            range_resolution: 距离分辨率 (m/bin)
            angle_resolution: 角度分辨率 (deg/bin)
            fft_size: FFT大小
            output_dir: 真实输出目录路径
            video_output_dir: 视频输出目录路径
            antenna_config_path: 天线配置路径
        """
        self.range_bins = range_bins
        self.angle_bins = angle_bins
        self.range_resolution = range_resolution
        self.angle_resolution = angle_resolution
        self.fft_size = fft_size
        self.output_dir = output_dir
        self.video_output_dir = video_output_dir
        self.antenna_config_path = antenna_config_path
        self.real_paths = REAL_PATHS
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # 预计算角度网格
        self.angle_grid = np.linspace(-90, 90, angle_bins)
        self.range_grid = np.arange(range_bins) * range_resolution
        
        # 加载天线配置
        self.antenna_positions = self._load_antenna_config()
        
        print(f"RA Processor初始化完成:")
        print(f"  输出目录: {self.output_dir}")
        print(f"  视频输出目录: {self.video_output_dir}")
        print(f"  天线配置路径: {self.antenna_config_path}")
        print(f"  天线位置形状: {self.antenna_positions.shape}")
    
    def _load_antenna_config(self) -> np.ndarray:
        """
        加载天线配置
        
        Returns:
            antenna_positions: 天线位置数组 [num_antennas, 2]
        """
        try:
            # 检查配置文件是否存在
            if os.path.exists(self.antenna_config_path):
                antenna_positions = np.load(self.antenna_config_path)
                print(f"加载天线配置: {self.antenna_config_path}")
                print(f"天线位置形状: {antenna_positions.shape}")
                return antenna_positions
            else:
                # 使用默认的均匀线阵配置
                print(f"天线配置文件不存在，使用默认配置: {self.antenna_config_path}")
                num_antennas = 8
                antenna_positions = np.zeros((num_antennas, 2))
                antenna_positions[:, 0] = np.arange(num_antennas) * 0.5  # 0.5米间距
                print(f"使用默认均匀线阵: {num_antennas} 个天线")
                return antenna_positions
        except Exception as e:
            print(f"加载天线配置失败: {e}")
            # 使用默认配置
            num_antennas = 8
            antenna_positions = np.zeros((num_antennas, 2))
            antenna_positions[:, 0] = np.arange(num_antennas) * 0.5
            return antenna_positions

    def range_fft(self, adc_data: np.ndarray) -> np.ndarray:
        """
        距离FFT处理
        
        Args:
            adc_data: ADC数据 [num_antennas, num_chirps, num_samples] 或 [tx, rx, num_chirps, num_samples]
            
        Returns:
            range_fft: 距离FFT结果 [num_antennas, num_chirps, range_bins]
        """
        # 处理不同形状的输入数据
        if len(adc_data.shape) == 4:
            # 形状为 (tx, rx, chirps, samples) - 合并tx和rx维度
            tx, rx, num_chirps, num_samples = adc_data.shape
            num_antennas = tx * rx
            # 重塑为 (num_antennas, num_chirps, num_samples)
            adc_data = adc_data.reshape(num_antennas, num_chirps, num_samples)
        elif len(adc_data.shape) == 3:
            # 形状为 (num_antennas, num_chirps, num_samples)
            num_antennas, num_chirps, num_samples = adc_data.shape
        else:
            raise ValueError(f"不支持的ADC数据形状: {adc_data.shape}，期望3D或4D")
        
        # 应用窗函数
        window = np.hanning(num_samples)
        windowed_data = adc_data * window.reshape(1, 1, -1)
        
        # 执行FFT
        range_fft = np.fft.fft(windowed_data, self.fft_size, axis=-1)
        range_fft = range_fft[:, :, :self.range_bins]
        
        print(f"距离FFT处理完成: {adc_data.shape} -> {range_fft.shape}")
        return range_fft
    
    def angle_fft(self, range_fft_data: np.ndarray, 
                  antenna_positions: np.ndarray) -> np.ndarray:
        """
        角度FFT处理
        
        Args:
            range_fft_data: 距离FFT数据 [num_antennas, num_chirps, range_bins]
            antenna_positions: 天线位置 [num_antennas, 2] (x, y坐标)
            
        Returns:
            ra_matrix: RA矩阵 [angle_bins, range_bins]
        """
        num_antennas, num_chirps, range_bins = range_fft_data.shape
        
        # 对每个距离单元进行角度FFT
        ra_matrix = np.zeros((self.angle_bins, range_bins), dtype=np.complex64)
        
        for r in range(range_bins):
            # 提取当前距离单元的相位信息
            range_slice = range_fft_data[:, :, r]
            
            # 计算协方差矩阵 - 确保正确的维度
            # 协方差矩阵应该是 (num_antennas, num_antennas)
            covariance = np.cov(range_slice)
            
            # 使用MUSIC算法进行角度估计
            angles, spectrum = self.music_algorithm(covariance, antenna_positions)
            
            # 将谱估计结果映射到RA矩阵
            for i, angle in enumerate(angles):
                angle_idx = int((angle + 90) / self.angle_resolution)
                if 0 <= angle_idx < self.angle_bins:
                    ra_matrix[angle_idx, r] = spectrum[i]
        
        return np.abs(ra_matrix)
    
    def music_algorithm(self, covariance: np.ndarray, 
                       antenna_positions: np.ndarray,
                       num_sources: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        MUSIC算法进行角度估计（实验性功能）
        
        Args:
            covariance: 协方差矩阵，形状必须为 (num_antennas, num_antennas)
            antenna_positions: 天线位置
            num_sources: 假设的信号源数量
            
        Returns:
            angles: 估计的角度 (度)
            spectrum: MUSIC谱
            
        Raises:
            NotImplementedError: 当空间维度不满足要求时抛出
        """
        # 检查输入维度
        if len(covariance.shape) != 2:
            raise NotImplementedError(f"MUSIC算法需要二维协方差矩阵，当前形状: {covariance.shape}")
        
        num_antennas = covariance.shape[0]
        if covariance.shape[1] != num_antennas:
            raise NotImplementedError(
                f"MUSIC算法需要方阵协方差矩阵，当前形状: {covariance.shape}")
        
        print("警告: MUSIC算法为实验性功能，可能不稳定")
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # 排序特征值
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 分离信号子空间和噪声子空间
        signal_subspace = eigenvectors[:, :num_sources]
        noise_subspace = eigenvectors[:, num_sources:]
        
        # 构建角度搜索网格
        search_angles = np.linspace(-90, 90, 360)
        spectrum = np.zeros(len(search_angles))
        
        # 计算MUSIC谱
        wavelength = 0.0039  # 77GHz波长
        for i, angle in enumerate(search_angles):
            # 构建导向矢量
            steering_vector = self._compute_steering_vector(
                antenna_positions, angle, wavelength)
            
            # 确保导向矢量形状正确 (num_antennas, 1)
            steering_vector = steering_vector.reshape(-1, 1)
            
            # 计算MUSIC谱 - 修复维度匹配问题
            # noise_subspace: (num_antennas, num_antennas - num_sources)
            # steering_vector: (num_antennas, 1)
            denominator = np.linalg.norm(noise_subspace.conj().T @ steering_vector) ** 2
            spectrum[i] = 1.0 / (denominator + 1e-10)
        
        return search_angles, spectrum

    def _compute_steering_vector(self, antenna_positions: np.ndarray,
                                angle: float, wavelength: float) -> np.ndarray:
        """
        计算导向矢量
        
        Args:
            antenna_positions: 天线位置
            angle: 角度 (度)
            wavelength: 波长
            
        Returns:
            steering_vector: 导向矢量
        """
        theta = np.deg2rad(angle)
        wave_number = 2 * np.pi / wavelength
        
        # 计算相位延迟
        phase_delays = wave_number * (antenna_positions[:, 0] * np.sin(theta) + 
                                     antenna_positions[:, 1] * np.cos(theta))
        
        steering_vector = np.exp(-1j * phase_delays)
        return steering_vector
    
    def generate_ra_matrix(self, adc_data: np.ndarray,
                          antenna_positions: np.ndarray) -> np.ndarray:
        """
        生成RA矩阵
        
        Args:
            adc_data: ADC数据
            antenna_positions: 天线位置
            
        Returns:
            ra_matrix: RA矩阵 [angle_bins, range_bins]
        """
        # 距离FFT
        range_fft = self.range_fft(adc_data)
        
        # 角度FFT
        ra_matrix = self.angle_fft(range_fft, antenna_positions)
        
        return ra_matrix
    
    def detect_targets(self, ra_matrix: np.ndarray, 
                      threshold_db: float = 10.0) -> List[dict]:
        """
        目标检测
        
        Args:
            ra_matrix: RA矩阵
            threshold_db: 检测阈值 (dB)
            
        Returns:
            targets: 检测到的目标列表
        """
        # 转换为dB
        ra_db = 20 * np.log10(ra_matrix + 1e-10)
        
        # 应用阈值
        threshold = np.max(ra_db) - threshold_db
        mask = ra_db > threshold
        
        # 寻找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8))
        
        targets = []
        for i in range(1, num_labels):  # 跳过背景
            # 计算目标属性
            angle_idx, range_idx = centroids[i]
            angle = self.angle_grid[int(angle_idx)]
            range_val = self.range_grid[int(range_idx)]
            intensity = np.max(ra_db[labels == i])
            
            targets.append({
                'angle': angle,
                'range': range_val,
                'intensity': intensity,
                'angle_idx': int(angle_idx),
                'range_idx': int(range_idx)
            })
        
        return targets
    
    def generate_ra_video(self, adc_sequence: List[np.ndarray],
                         antenna_positions: np.ndarray,
                         output_path: str) -> None:
        """
        生成RA视频序列
        
        Args:
            adc_sequence: ADC数据序列
            antenna_positions: 天线位置
            output_path: 输出路径
        """
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            output_path, fourcc, 10.0, 
            (self.angle_bins, self.range_bins), isColor=False)
        
        for adc_data in adc_sequence:
            # 生成RA矩阵
            ra_matrix = self.generate_ra_matrix(adc_data, antenna_positions)
            
            # 转换为8位图像
            ra_normalized = (ra_matrix / np.max(ra_matrix) * 255).astype(np.uint8)
            
            # 写入视频帧
            video_writer.write(ra_normalized)
        
        video_writer.release()
    
    def plot_ra_matrix(self, ra_matrix: np.ndarray, 
                      title: str = "RA Matrix",
                      save_path: Optional[str] = None,
                      sequence_name: str = "test_sequence") -> str:
        """
        绘制RA矩阵
        
        Args:
            ra_matrix: RA矩阵
            title: 图像标题
            save_path: 保存路径（可选）
            sequence_name: 序列名称，用于自动生成保存路径
            
        Returns:
            save_path: 实际保存路径
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(20 * np.log10(ra_matrix + 1e-10), 
                  extent=[0, self.range_bins * self.range_resolution, 
                         -90, 90],
                  aspect='auto', cmap='jet')
        plt.colorbar(label='Intensity (dB)')
        plt.xlabel('Range (m)')
        plt.ylabel('Angle (deg)')
        plt.title(title)
        plt.tight_layout()
        
        # 自动生成保存路径
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"ra_matrix_{sequence_name}_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存RA图: {save_path}")
        
        plt.show()
        
        return save_path


def test_ra_processing():
    """测试RA处理功能"""
    # 创建测试数据
    processor = RAProcessor()
    
    # 模拟天线位置 (线性阵列)
    num_antennas = 8
    antenna_positions = np.zeros((num_antennas, 2))
    antenna_positions[:, 0] = np.arange(num_antennas) * 0.5  # 0.5波长间距
    
    # 模拟ADC数据
    num_chirps = 64
    num_samples = 256
    adc_data = np.random.randn(num_antennas, num_chirps, num_samples) + \
               1j * np.random.randn(num_antennas, num_chirps, num_samples)
    
    # 生成RA矩阵
    ra_matrix = processor.generate_ra_matrix(adc_data, antenna_positions)
    
    # 目标检测
    targets = processor.detect_targets(ra_matrix)
    print(f"检测到 {len(targets)} 个目标")
    
    # 绘制RA矩阵
    processor.plot_ra_matrix(ra_matrix, "测试RA矩阵")
    
    return processor, ra_matrix, targets


if __name__ == "__main__":
    test_ra_processing()