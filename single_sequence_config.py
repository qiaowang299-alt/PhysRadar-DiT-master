"""
单序列训练配置文件
专门针对 2_28_2021_outdoors_run9 序列优化
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class SingleSequenceTrainingConfig:
    """单序列训练配置"""
    data_root: str = "d:/桌面/PhysRadar-DiT-master/RaDar/coloradar/dataset"
    sequence_name: str = "2_28_2021_outdoors_run9"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    num_epochs: int = 20
    learning_rate: float = 1e-4
    input_size: int = 256
    checkpoint_dir: str = "checkpoints/single_sequence"
    log_dir: str = "logs/single_sequence"
    use_complex_vae: bool = True
    use_controlnet: bool = True

@dataclass
class SingleSequenceDataConfig:
    """单序列数据配置"""
    radar_type: str = "adc"
    chirps: int = 64
    samples: int = 256
    num_frames: int = 793
    train_split: float = 0.8
    val_split: float = 0.15
    test_split: float = 0.05

@dataclass
class SingleSequenceVQVAEConfig:
    """VQ-VAE配置"""
    latent_dim: int = 16
    codebook_size: int = 512
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)
    use_complex: bool = True

@dataclass
class SingleSequenceDiTConfig:
    """DiT配置"""
    input_channels: int = 16
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    patch_size: int = 2

@dataclass
class SingleSequenceControlNetConfig:
    """ControlNet配置"""
    control_channels: int = 3
    control_scale: float = 1.0

def get_single_sequence_config():
    """获取单序列训练配置"""
    import torch
    
    return {
        'training': SingleSequenceTrainingConfig(),
        'data': SingleSequenceDataConfig(),
        'vqvae': SingleSequenceVQVAEConfig(),
        'dit': SingleSequenceDiTConfig(),
        'controlnet': SingleSequenceControlNetConfig()
    }