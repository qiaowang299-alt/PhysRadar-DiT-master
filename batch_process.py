"""
批量处理模块
实现雷达数据的批量处理和分析
"""

import numpy as np
import torch
import multiprocessing as mp
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Callable
import os
import glob
from tqdm import tqdm
import pandas as pd

from .adc_reader import ADCReader
from .rd_processing import RDProcessor
from .ra_processing import RAProcessor
from .visualize import RadarVisualizer

# 真实物理地址配置
REAL_PATHS = {
    # 项目相关路径
    "PROJECT_ROOT": r"D:\桌面\PhysRadar-DiT-master",
    "RADAR_PROCESSING_DIR": r"D:\桌面\PhysRadar-DiT-master\radar_processing",
    "OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output",
    "BATCH_OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output\batch_processing",
    "PROCESSED_RESULTS_DIR": r"D:\桌面\PhysRadar-DiT-master\output\processed_results",
    "BATCH_PROCESSED_DIR": r"D:\桌面\PhysRadar-DiT-master\output\batch_processed",
    "MODEL_DIR": r"D:\桌面\PhysRadar-DiT-master\models",
    "CONFIG_DIR": r"D:\桌面\PhysRadar-DiT-master\configs",
    
    # 数据集路径
    "COLO_RADAR_DATASET_PATH": r"D:\datasets\CoLoRadar",
    "RADAR_DATA_DIR": r"D:\datasets\radar_data",
    "TEST_DATA_DIR": r"D:\datasets\test_data",
    
    # 默认配置路径
    "DEFAULT_CONFIG": r"D:\桌面\PhysRadar-DiT-master\configs\batch_processing_config.json"
}


class BatchProcessor:
    """
    批量处理器 - 处理大量雷达数据
    """
    
    def __init__(self, 
                 config_path: Optional[str] = REAL_PATHS["DEFAULT_CONFIG"],
                 num_workers: int = 4,
                 output_dir: str = REAL_PATHS["BATCH_OUTPUT_DIR"],
                 data_dir: str = REAL_PATHS["COLO_RADAR_DATASET_PATH"]):
        """
        初始化批量处理器
        
        Args:
            config_path: 真实配置文件路径
            num_workers: 并行工作进程数
            output_dir: 真实输出目录路径
            data_dir: 真实数据目录路径
        """
        self.num_workers = min(num_workers, mp.cpu_count())
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.real_paths = REAL_PATHS
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化处理器
        self.adc_reader = ADCReader()
        self.rd_processor = RDProcessor()
        self.ra_processor = RAProcessor()
        self.visualizer = RadarVisualizer()
        
        # 设置日志
        self._setup_logging()
        
        print(f"Batch Processor初始化完成:")
        print(f"  数据目录: {self.data_dir}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  工作进程数: {self.num_workers}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        default_config = {
            'range_bins': 256,
            'angle_bins': 180,
            'doppler_bins': 64,
            'range_resolution': 0.1,
            'angle_resolution': 1.0,
            'doppler_resolution': 0.1,
            'processing_steps': ['adc_reading', 'rd_processing', 'ra_processing'],
            'save_formats': ['npy', 'png', 'csv']
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'batch_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def find_data_files(self, 
                       data_dir: Optional[str] = None,
                       patterns: List[str] = ['*.bin', '*.npy', '*.mat']) -> List[Path]:
        """
        查找数据文件
        
        Args:
            data_dir: 真实数据目录路径
            patterns: 文件模式列表
            
        Returns:
            data_files: 数据文件列表
        """
        if data_dir is None:
            data_path = self.data_dir
        else:
            data_path = Path(data_dir)
        
        data_files = []
        
        for pattern in patterns:
            files = list(data_path.rglob(pattern))
            data_files.extend(files)
        
        # 去重并排序
        data_files = sorted(list(set(data_files)))
        
        self.logger.info(f"在 {data_path} 中找到 {len(data_files)} 个数据文件")
        print(f"数据文件搜索完成:")
        print(f"  搜索目录: {data_path}")
        print(f"  找到文件数: {len(data_files)}")
        
        return data_files
    
    def process_single_file(self, file_path: Path, 
                          output_subdir: str = "processed") -> Dict:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            output_subdir: 输出子目录
            
        Returns:
            result: 处理结果字典
        """
        try:
            start_time = time.time()
            
            # 创建输出目录
            output_dir = self.output_dir / output_subdir / file_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = {
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'processing_time': 0,
                'success': True,
                'error_message': None,
                'output_files': []
            }
            
            # 1. 读取ADC数据
            self.logger.info(f"处理文件: {file_path}")
            adc_data = self.adc_reader.read_binary_file(str(file_path))
            
            if adc_data is None:
                raise ValueError(f"无法读取ADC数据: {file_path}")
            
            # 2. 检查ADC数据合法性
            self._validate_adc_data(adc_data)
            
            # 3. 预处理ADC数据以适应RD处理
            processed_adc_data = self._preprocess_adc_data(adc_data)
            
            # 4. RD处理
            rd_matrix = self.rd_processor.generate_rd_matrix(processed_adc_data)
            
            # 5. RA处理 (需要天线位置信息)
            # 这里使用默认的天线位置
            antenna_positions = np.zeros((8, 2))
            antenna_positions[:, 0] = np.arange(8) * 0.5
            
            ra_matrix = self.ra_processor.generate_ra_matrix(adc_data, antenna_positions)
            
            # 6. 保存结果
            output_files = self._save_processing_results(
                output_dir, adc_data, rd_matrix, ra_matrix)
            
            result['output_files'] = output_files
            result['processing_time'] = time.time() - start_time
            
            self.logger.info(f"文件处理完成: {file_path} (耗时: {result['processing_time']:.2f}s)")
            
        except Exception as e:
            result['success'] = False
            result['error_message'] = str(e)
            result['processing_time'] = time.time() - start_time
            self.logger.error(f"处理文件失败: {file_path} - {e}")
        
        return result
    
    def _validate_adc_data(self, adc_data: np.ndarray) -> None:
        """
        验证ADC数据合法性
        
        Args:
            adc_data: ADC数据
            
        Raises:
            ValueError: 当数据不合法时抛出
        """
        # 检查是否为numpy数组
        if not isinstance(adc_data, np.ndarray):
            raise ValueError(f"ADC数据必须为numpy数组，当前类型: {type(adc_data)}")
        
        # 检查维度
        if len(adc_data.shape) not in [3, 4]:
            raise ValueError(f"ADC数据维度必须为3或4，当前形状: {adc_data.shape}")
        
        # 允许的形状：(Tx, Rx, chirps, samples) 或 (Rx, chirps, samples)
        if len(adc_data.shape) == 4:
            # (Tx, Rx, chirps, samples)
            tx, rx, chirps, samples = adc_data.shape
            if tx <= 0 or rx <= 0 or chirps <= 0 or samples <= 0:
                raise ValueError(f"ADC数据维度必须为正数，当前形状: {adc_data.shape}")
        else:
            # (Rx, chirps, samples)
            rx, chirps, samples = adc_data.shape
            if rx <= 0 or chirps <= 0 or samples <= 0:
                raise ValueError(f"ADC数据维度必须为正数，当前形状: {adc_data.shape}")
        
        self.logger.info(f"ADC数据验证通过，形状: {adc_data.shape}")
    
    def _preprocess_adc_data(self, adc_data: np.ndarray) -> np.ndarray:
        """
        预处理ADC数据以适应RD处理
        
        Args:
            adc_data: 原始ADC数据
            
        Returns:
            processed_data: 处理后的数据，形状为 (chirps, samples)
        """
        # 根据输入形状选择处理方式
        if len(adc_data.shape) == 4:
            # (Tx, Rx, chirps, samples) -> 选择第一个Tx和Rx，然后合并chirps
            processed_data = adc_data[0, 0]  # 选择第一个Tx和Rx
        else:
            # (Rx, chirps, samples) -> 选择第一个Rx
            processed_data = adc_data[0]  # 选择第一个Rx
        
        # 确保输出形状为 (chirps, samples)
        if len(processed_data.shape) != 2:
            raise ValueError(f"预处理后数据形状必须为2维，当前形状: {processed_data.shape}")
        
        self.logger.info(f"ADC数据预处理完成，输出形状: {processed_data.shape}")
        return processed_data
    
    def _save_processing_results(self, output_dir: Path,
                               adc_data: np.ndarray,
                               rd_matrix: np.ndarray,
                               ra_matrix: np.ndarray) -> List[str]:
        """
        保存处理结果
        
        Returns:
            saved_files: 保存的文件列表
        """
        saved_files = []
        
        # 保存原始数据
        if 'npy' in self.config['save_formats']:
            np.save(output_dir / 'adc_data.npy', adc_data)
            np.save(output_dir / 'rd_matrix.npy', rd_matrix)
            np.save(output_dir / 'ra_matrix.npy', ra_matrix)
            saved_files.extend(['adc_data.npy', 'rd_matrix.npy', 'ra_matrix.npy'])
        
        # 保存图像
        if 'png' in self.config['save_formats']:
            self.visualizer.plot_rd_matrix(rd_matrix, 
                                         save_path=str(output_dir / 'rd_matrix.png'))
            self.visualizer.plot_ra_matrix(ra_matrix,
                                         save_path=str(output_dir / 'ra_matrix.png'))
            saved_files.extend(['rd_matrix.png', 'ra_matrix.png'])
        
        # 保存统计信息
        if 'csv' in self.config['save_formats']:
            stats = self._compute_statistics(adc_data, rd_matrix, ra_matrix)
            stats_df = pd.DataFrame([stats])
            stats_df.to_csv(output_dir / 'statistics.csv', index=False)
            saved_files.append('statistics.csv')
        
        return saved_files
    
    def _compute_statistics(self, adc_data: np.ndarray,
                          rd_matrix: np.ndarray,
                          ra_matrix: np.ndarray) -> Dict:
        """计算统计信息"""
        return {
            'adc_mean': np.mean(adc_data),
            'adc_std': np.std(adc_data),
            'adc_max': np.max(adc_data),
            'adc_min': np.min(adc_data),
            'rd_mean': np.mean(rd_matrix),
            'rd_std': np.std(rd_matrix),
            'rd_max': np.max(rd_matrix),
            'ra_mean': np.mean(ra_matrix),
            'ra_std': np.std(ra_matrix),
            'ra_max': np.max(ra_matrix),
            'num_targets_rd': len(self.rd_processor.detect_targets(rd_matrix)),
            'num_targets_ra': len(self.ra_processor.detect_targets(ra_matrix))
        }
    
    def process_batch(self, 
                     data_files: Optional[List[Path]] = None,
                     data_dir: Optional[str] = None,
                     output_subdir: str = "batch_processed") -> pd.DataFrame:
        """
        批量处理文件
        
        Args:
            data_files: 数据文件列表 (如果为None则自动搜索)
            data_dir: 真实数据目录路径
            output_subdir: 输出子目录
            
        Returns:
            results_df: 处理结果数据框
        """
        # 自动搜索数据文件
        if data_files is None:
            data_files = self.find_data_files(data_dir)
        
        if not data_files:
            self.logger.warning("未找到任何数据文件")
            return pd.DataFrame()
        
        self.logger.info(f"开始批量处理 {len(data_files)} 个文件")
        print(f"开始批量处理:")
        print(f"  文件数量: {len(data_files)}")
        print(f"  输出目录: {self.output_dir / output_subdir}")
        
        results = []
        
        # 使用多进程处理
        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                process_func = lambda f: self.process_single_file(f, output_subdir)
                for result in tqdm(pool.imap(process_func, data_files), 
                                 total=len(data_files)):
                    results.append(result)
        else:
            # 单进程处理
            for file_path in tqdm(data_files):
                result = self.process_single_file(file_path, output_subdir)
                results.append(result)
        
        # 生成统计报告
        report = self._generate_report(results)
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_path = self.output_dir / 'processing_results.csv'
        results_df.to_csv(results_path, index=False)
        
        report_path = self.output_dir / 'processing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"批量处理完成，成功处理 {report['success_count']}/{len(data_files)} 个文件")
        print(f"批量处理完成:")
        print(f"  成功文件数: {report['success_count']}/{len(data_files)}")
        print(f"  成功率: {report['success_rate']:.2%}")
        print(f"  平均处理时间: {report['avg_processing_time']:.2f}s")
        print(f"  结果文件: {results_path}")
        print(f"  报告文件: {report_path}")
        
        return results_df
    
    def _generate_report(self, results: List[Dict]) -> Dict:
        """生成处理报告"""
        success_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - success_count
        
        processing_times = [r['processing_time'] for r in results if r['success']]
        file_sizes = [r['file_size'] for r in results if r['success']]
        
        return {
            'total_files': len(results),
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_count / len(results) if len(results) > 0 else 0,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'max_processing_time': np.max(processing_times) if processing_times else 0,
            'min_processing_time': np.min(processing_times) if processing_times else 0,
            'avg_file_size': np.mean(file_sizes) if file_sizes else 0,
            'failed_files': [r['file_path'] for r in results if not r['success']]
        }
    
    def analyze_processing_results(self, results_df: pd.DataFrame) -> Dict:
        """
        分析处理结果
        
        Args:
            results_df: 处理结果数据框
            
        Returns:
            analysis: 分析结果
        """
        analysis = {}
        
        # 基本统计
        analysis['total_files'] = len(results_df)
        analysis['success_rate'] = results_df['success'].mean()
        
        # 处理时间分析
        success_results = results_df[results_df['success']]
        if not success_results.empty:
            analysis['processing_time_stats'] = {
                'mean': success_results['processing_time'].mean(),
                'std': success_results['processing_time'].std(),
                'min': success_results['processing_time'].min(),
                'max': success_results['processing_time'].max()
            }
            
            # 文件大小与处理时间的关系
            analysis['size_time_correlation'] = success_results[['file_size', 'processing_time']].corr().iloc[0, 1]