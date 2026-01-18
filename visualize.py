"""
雷达数据可视化模块
实现雷达数据的多种可视化方式
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import cv2
import torch
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
from datetime import datetime

# 真实物理地址配置
REAL_PATHS = {
    # 项目相关路径
    "PROJECT_ROOT": r"D:\桌面\PhysRadar-DiT-master",
    "RADAR_PROCESSING_DIR": r"D:\桌面\PhysRadar-DiT-master\radar_processing",
    "OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output",
    "VISUALIZATION_OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output\visualizations",
    "ANIMATION_OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output\animations",
    "COMPARISON_OUTPUT_DIR": r"D:\桌面\PhysRadar-DiT-master\output\comparisons",
    "MODEL_DIR": r"D:\桌面\PhysRadar-DiT-master\models",
    
    # 预训练模型路径
    "PRETRAINED_MODELS": {
        "visualizer": r"D:\桌面\PhysRadar-DiT-master\models\radar_visualizer.pth"
    },
    
    # 默认数据集路径
    "DEFAULT_DATASET": r"D:\datasets\CoLoRadar"
}


class RadarVisualizer:
    """
    雷达数据可视化器
    """
    
    def __init__(self, 
                 range_bins: int = 256,
                 angle_bins: int = 180,
                 doppler_bins: int = 64,
                 range_resolution: float = 0.1,
                 angle_resolution: float = 1.0,
                 doppler_resolution: float = 0.1,
                 output_dir: str = REAL_PATHS["VISUALIZATION_OUTPUT_DIR"],
                 animation_dir: str = REAL_PATHS["ANIMATION_OUTPUT_DIR"],
                 comparison_dir: str = REAL_PATHS["COMPARISON_OUTPUT_DIR"]):
        """
        初始化可视化器
        
        Args:
            range_bins: 距离维度大小
            angle_bins: 角度维度大小
            doppler_bins: 多普勒维度大小
            range_resolution: 距离分辨率 (m/bin)
            angle_resolution: 角度分辨率 (deg/bin)
            doppler_resolution: 多普勒分辨率 (m/s/bin)
            output_dir: 真实输出目录路径
            animation_dir: 动画输出目录路径
            comparison_dir: 对比图输出目录路径
        """
        self.range_bins = range_bins
        self.angle_bins = angle_bins
        self.doppler_bins = doppler_bins
        self.range_resolution = range_resolution
        self.angle_resolution = angle_resolution
        self.doppler_resolution = doppler_resolution
        self.output_dir = output_dir
        self.animation_dir = animation_dir
        self.comparison_dir = comparison_dir
        self.real_paths = REAL_PATHS
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.animation_dir, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('dark_background')
        
        print(f"Radar Visualizer初始化完成:")
        print(f"  可视化输出目录: {self.output_dir}")
        print(f"  动画输出目录: {self.animation_dir}")
        print(f"  对比图输出目录: {self.comparison_dir}")
    
    def plot_rd_matrix(self, rd_matrix: np.ndarray, 
                      title: str = "RD Matrix",
                      save_path: Optional[str] = None,
                      sequence_name: str = "test_sequence") -> plt.Figure:
        """
        绘制RD矩阵
        
        Args:
            rd_matrix: RD矩阵 [doppler_bins, range_bins]
            title: 图像标题
            save_path: 保存路径
            sequence_name: 序列名称，用于自动生成保存路径
            
        Returns:
            fig: matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 转换为dB
        rd_db = 20 * np.log10(rd_matrix + 1e-10)
        
        # 绘制热图
        im = ax.imshow(rd_db, 
                      extent=[0, self.range_bins * self.range_resolution,
                             -self.doppler_bins//2 * self.doppler_resolution,
                             self.doppler_bins//2 * self.doppler_resolution],
                      aspect='auto', cmap='jet', origin='lower')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity (dB)')
        
        # 设置坐标轴标签
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Doppler Velocity (m/s)')
        ax.set_title(title)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 自动生成保存路径
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"rd_matrix_{sequence_name}_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RD矩阵图已保存到: {save_path}")
        
        return fig
    
    def plot_ra_matrix(self, ra_matrix: np.ndarray,
                      title: str = "RA Matrix",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制RA矩阵
        
        Args:
            ra_matrix: RA矩阵 [angle_bins, range_bins]
            title: 图像标题
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 转换为dB
        ra_db = 20 * np.log10(ra_matrix + 1e-10)
        
        # 绘制热图
        im = ax.imshow(ra_db,
                      extent=[0, self.range_bins * self.range_resolution,
                             -90, 90],
                      aspect='auto', cmap='jet')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity (dB)')
        
        # 设置坐标轴标签
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Angle (deg)')
        ax.set_title(title)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"RA矩阵图已保存到: {save_path}")
        
        return fig
    
    def plot_3d_pointcloud(self, pointcloud: np.ndarray,
                          title: str = "3D Point Cloud",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制3D点云
        
        Args:
            pointcloud: 点云数据 [N, 4] (x, y, z, intensity)
            title: 图像标题
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图形对象
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 创建3D子图
        ax = fig.add_subplot(111, projection='3d')
        
        if pointcloud.shape[1] >= 4:
            # 使用强度作为颜色
            scatter = ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
                               c=pointcloud[:, 3], cmap='viridis', s=10, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Intensity')
        else:
            ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
                      s=10, alpha=0.7)
        
        # 设置坐标轴标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 设置相等的比例尺
        max_range = np.max(np.abs(pointcloud[:, :3]))
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D点云图已保存到: {save_path}")
        
        return fig
    
    def plot_bev(self, pointcloud: np.ndarray,
                title: str = "Bird's Eye View",
                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制鸟瞰图
        
        Args:
            pointcloud: 点云数据 [N, 4] (x, y, z, intensity)
            title: 图像标题
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if pointcloud.shape[1] >= 4:
            # 使用强度作为颜色
            scatter = ax.scatter(pointcloud[:, 0], pointcloud[:, 1],
                               c=pointcloud[:, 3], cmap='viridis', s=20, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Intensity')
        else:
            ax.scatter(pointcloud[:, 0], pointcloud[:, 1], s=20, alpha=0.7)
        
        # 设置坐标轴标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        
        # 设置网格和比例尺
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 设置坐标轴范围
        max_range = np.max(np.abs(pointcloud[:, :2]))
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"鸟瞰图已保存到: {save_path}")
        
        return fig
    
    def create_animation(self, matrix_sequence: List[np.ndarray],
                        output_path: str,
                        matrix_type: str = 'rd',
                        fps: int = 10,
                        sequence_name: str = "test_sequence") -> str:
        """
        创建雷达数据动画
        
        Args:
            matrix_sequence: 矩阵序列
            output_path: 输出路径
            matrix_type: 矩阵类型 ('rd', 'ra')
            fps: 帧率
            sequence_name: 序列名称，用于自动生成保存路径
            
        Returns:
            output_path: 实际保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 自动生成保存路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.animation_dir, f"{matrix_type}_animation_{sequence_name}_{timestamp}.mp4")
        
        def update(frame):
            ax.clear()
            matrix = matrix_sequence[frame]
            
            if matrix_type == 'rd':
                matrix_db = 20 * np.log10(matrix + 1e-10)
                im = ax.imshow(matrix_db, aspect='auto', cmap='jet', origin='lower')
                ax.set_xlabel('Range (m)')
                ax.set_ylabel('Doppler Velocity (m/s)')
                ax.set_title(f'RD Matrix - Frame {frame}')
            else:  # 'ra'
                matrix_db = 20 * np.log10(matrix + 1e-10)
                im = ax.imshow(matrix_db, aspect='auto', cmap='jet')
                ax.set_xlabel('Range (m)')
                ax.set_ylabel('Angle (deg)')
                ax.set_title(f'RA Matrix - Frame {frame}')
            
            return [im]
        
        # 创建动画
        anim = animation.FuncAnimation(fig, update, frames=len(matrix_sequence),
                                     interval=1000//fps, blit=True)
        
        # 保存动画
        anim.save(output_path, writer='ffmpeg', fps=fps)
        print(f"动画已保存到: {output_path}")
        
        plt.close(fig)
        
        return output_path
    
    def plot_comparison(self, real_data: np.ndarray, 
                       generated_data: np.ndarray,
                       titles: Tuple[str, str] = ('Real Data', 'Generated Data'),
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制真实数据与生成数据的对比
        
        Args:
            real_data: 真实数据
            generated_data: 生成数据
            titles: 标题元组
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图形对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 绘制真实数据
        real_db = 20 * np.log10(real_data + 1e-10)
        im1 = ax1.imshow(real_db, aspect='auto', cmap='jet')
        ax1.set_title(titles[0])
        plt.colorbar(im1, ax=ax1)
        
        # 绘制生成数据
        gen_db = 20 * np.log10(generated_data + 1e-10)
        im2 = ax2.imshow(gen_db, aspect='auto', cmap='jet')
        ax2.set_title(titles[1])
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存到: {save_path}")
        
        return fig
    
    def save_radar_image(self, matrix: np.ndarray, 
                        output_path: str,
                        normalize: bool = True) -> None:
        """
        保存雷达图像
        
        Args:
            matrix: 雷达矩阵
            output_path: 输出路径
            normalize: 是否归一化
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if normalize:
            # 归一化到0-255
            matrix_normalized = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
            image = (matrix_normalized * 255).astype(np.uint8)
        else:
            image = matrix.astype(np.uint8)
        
        # 保存图像
        cv2.imwrite(output_path, image)
        print(f"雷达图像已保存到: {output_path}")


def test_visualization():
    """测试可视化功能"""
    # 创建测试数据
    visualizer = RadarVisualizer()
    
    # 生成测试RD矩阵
    rd_matrix = np.random.rand(64, 256)
    
    # 测试各种可视化功能
    visualizer.plot_rd_matrix(rd_matrix, "测试RD矩阵")
    
    # 生成测试RA矩阵
    ra_matrix = np.random.rand(180, 256)
    visualizer.plot_ra_matrix(ra_matrix, "测试RA矩阵")
    
    # 生成测试点云
    pointcloud = np.random.randn(1000, 4)
    pointcloud[:, 3] = np.abs(pointcloud[:, 3])  # 强度为正
    
    visualizer.plot_3d_pointcloud(pointcloud, "测试3D点云")
    visualizer.plot_bev(pointcloud, "测试鸟瞰图")
    
    print("可视化测试完成")


if __name__ == "__main__":
    test_visualization()