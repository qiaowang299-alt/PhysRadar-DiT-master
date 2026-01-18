"""
ColoRadar训练配置文件
"""

from dataclasses import dataclass
from .base_config import BaseConfig, RadarDataConfig, VQVAEConfig, DiTConfig, ControlNetConfig


@dataclass
class ColoRadarTrainingConfig(BaseConfig):
    """ColoRadar训练配置"""
    
    # 数据配置 - 修改为你的实际路径
    data_root: str = "d:/桌面/PhysRadar-DiT-master/RaDar/coloradar/dataset"  # 你的ColoRadar数据路径
    sequences: tuple = ("2_28_2021_outdoors_run9",)  # 使用你已有的序列
    input_size: tuple = (128, 128)  # 输入尺寸
    
    # 训练参数
    batch_size: int = 4  # 小批次以适应内存
    num_epochs: int = 10  # 初始训练轮数
    learning_rate: float = 1e-4
    
    # 模型配置
    use_complex_vae: bool = True
    use_controlnet: bool = True
    
    # 保存配置
    save_every: int = 5  # 每5轮保存一次
    eval_every: int = 2  # 每2轮评估一次


# ColoRadar特定的雷达配置
@dataclass  
class ColoRadarDataConfig(RadarDataConfig):
    """ColoRadar数据配置"""
    
    # 根据ColoRadar特性调整
    num_chirps: int = 64    # AWR1843的方位角分辨率
    num_samples: int = 256  # 距离分辨率
    num_antennas: int = 3   # AWR1843天线配置
    
    # 数据增强
    noise_std: float = 0.05  # 增加噪声以模拟真实环境
    random_phase_shift: bool = True


# 针对ColoRadar优化的模型配置
@dataclass
class ColoRadarVQVAEConfig(VQVAEConfig):
    """ColoRadar VQ-VAE配置"""
    
    # 较小的模型以适应ColoRadar数据
    latent_dim: int = 8
    latent_h: int = 32
    latent_w: int = 32
    codebook_size: int = 256
    
    # 更强的相位保留
    phase_loss_weight: float = 1.0


@dataclass
class ColoRadarDiTConfig(DiTConfig):
    """ColoRadar DiT配置"""
    
    # 较小的模型
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 8
    
    # 更快的扩散过程
    num_diffusion_steps: int = 500


def get_colo_radar_config() -> dict:
    """获取完整的ColoRadar配置"""
    return {
        'training': ColoRadarTrainingConfig(),
        'data': ColoRadarDataConfig(),
        'vqvae': ColoRadarVQVAEConfig(),
        'dit': ColoRadarDiTConfig(),
        'controlnet': ControlNetConfig()
    }