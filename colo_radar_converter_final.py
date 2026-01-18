"""
ColoRadar数据转换器 - 最终版本
利用完整的ColoRadar库处理原始ADC数据并转换为PhysRadar-DiT训练格式
"""

import os
import sys
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import h5py

# 添加ColoRadar库路径
sys.path.insert(0, str(Path(__file__).parent.parent / "RaDar" / "coloradar"))

from core.dataset import Coloradar
from core.record import Record
from core.radar import SCRadar, CCRadar
from core.config import *


class ColoRadarConverter:
    """ColoRadar数据转换器 - 利用完整ColoRadar库"""
    
    def __init__(self, dataset_root: str = None, output_dir: str = "./processed_data"):
        """
        初始化转换器
        
        Args:
            dataset_root: ColoRadar数据集根目录（如果为None，使用默认路径）
            output_dir: 处理后的数据保存目录
        """
        if dataset_root is None:
            # 使用你复制的ColoRadar数据集路径
            self.dataset_root = Path(__file__).parent.parent / "RaDar" / "coloradar" / "dataset"
        else:
            self.dataset_root = Path(dataset_root)
            
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化ColoRadar数据集
        print("初始化ColoRadar数据集...")
        self.coloradar = Coloradar(str(self.dataset_root))
        
        # 打印可用的数据集
        self.coloradar.printCodenames()
        
        # 数据参数
        self.radar_params = {
            'range_bins': NUMBER_RANGE_BINS_MIN,  # 128
            'azimuth_bins': NUMBER_AZIMUTH_BINS_MIN,  # 32
            'elevation_bins': NUMBER_ELEVATION_BINS_MIN,  # 64
            'doppler_bins': NUMBER_DOPPLER_BINS_MIN,  # 16
            'doa_method': DOA_METHOD,  # "esprit"
            'rdsp_method': RDSP_METHOD,  # "normal"
        }
        
        print(f"雷达参数: {self.radar_params}")
    
    def get_sequence_info(self, codename: str) -> Dict:
        """获取序列信息"""
        # 查找序列信息
        for dataset in self.coloradar.config["datastore"]["folders"]:
            if dataset["codename"] == codename:
                return dataset
        
        raise ValueError(f"序列 {codename} 不存在")
    
    def load_radar_frame(self, codename: str, frame_idx: int) -> Dict:
        """
        加载单帧雷达数据
        
        Args:
            codename: 序列代号
            frame_idx: 帧索引
            
        Returns:
            包含雷达数据的字典
        """
        try:
            # 使用ColoRadar库加载记录
            record = self.coloradar.getRecord(codename, frame_idx)
            
            radar_data = {}
            
            # 检查雷达类型（单芯片或级联）
            if hasattr(record, 'ccradar') and record.ccradar is not None:
                radar = record.ccradar
                radar_type = "ccradar"
            elif hasattr(record, 'scradar') and record.scradar is not None:
                radar = record.scradar
                radar_type = "scradar"
            else:
                print(f"帧 {frame_idx}: 无雷达数据")
                return {}
            
            # 获取原始ADC数据
            if radar.raw is not None:
                radar_data['adc_raw'] = radar.raw
                radar_data['adc_shape'] = radar.raw.shape
                print(f"帧 {frame_idx}: ADC数据形状 {radar.raw.shape}")
            
            # 获取热图数据
            if radar.heatmap is not None:
                radar_data['heatmap'] = radar.heatmap
                print(f"帧 {frame_idx}: 热图数据形状 {radar.heatmap.shape}")
            
            # 获取点云数据
            if radar.cld is not None and len(radar.cld) > 0:
                radar_data['pointcloud'] = radar.cld
                print(f"帧 {frame_idx}: 点云数据点数 {len(radar.cld)}")
            
            radar_data['radar_type'] = radar_type
            return radar_data
            
        except Exception as e:
            print(f"加载帧 {frame_idx} 失败: {e}")
            return {}
    
    def process_adc_to_complex_rd(self, adc_data: np.ndarray, radar_type: str) -> torch.Tensor:
        """
        处理ADC数据生成复数Range-Doppler Map
        
        Args:
            adc_data: 原始ADC数据
            radar_type: 雷达类型
            
        Returns:
            complex_rd_map: (2, H, W) 复数RD Map (实部+虚部)
        """
        try:
            if radar_type == "ccradar":
                # 级联雷达数据处理
                # 形状: (tx, rx, chirps, samples)
                tx, rx, chirps, samples = adc_data.shape
                
                # 对每个接收通道进行处理
                rd_maps = []
                for r in range(rx):
                    # 对单个接收通道的数据进行2D FFT
                    channel_data = adc_data[0, r, :, :]  # 使用第一个发射天线
                    
                    # 距离FFT
                    range_fft = np.fft.fft(channel_data, axis=1)
                    
                    # 多普勒FFT
                    doppler_fft = np.fft.fft(range_fft, axis=0)
                    
                    # 转换为复数RD Map
                    rd_map = doppler_fft
                    rd_maps.append(rd_map)
                
                # 合并所有通道（简单平均）
                combined_rd = np.mean(rd_maps, axis=0)
                
            else:  # scradar
                # 单芯片雷达数据处理
                # 形状: (tx, rx, chirps, samples)
                tx, rx, chirps, samples = adc_data.shape
                
                # 对单个通道进行2D FFT
                channel_data = adc_data[0, 0, :, :]  # 使用第一个天线对
                
                # 距离FFT
                range_fft = np.fft.fft(channel_data, axis=1)
                
                # 多普勒FFT
                doppler_fft = np.fft.fft(range_fft, axis=0)
                
                combined_rd = doppler_fft
            
            # 分离实部和虚部
            real_part = np.real(combined_rd)