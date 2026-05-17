import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    # ========================
    # Random seed / Device
    # ========================
    SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========================
    # Пути до файлов/директорий
    # ========================
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    
    IMAGE_DIR: Path = BASE_DIR / 'data' / 'images'
    PROCESSED_DATA_PATH: Path = BASE_DIR / 'data' / 'processed' / 'dish_processed.csv'
    SAVE_MODEL_PATH: Path = BASE_DIR / 'models' / 'best_model.pth'
    
    # ========================
    # Модели
    # ========================
    TEXT_MODEL_NAME: str = 'bert-base-uncased'
    IMAGE_MODEL_NAME: str = 'efficientnet_b0'
    
    # ========================
    # Данные
    # ========================
    MAX_LENGTH: int = 64
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = min(8, os.cpu_count() or 0)
    
    # ========================
    # Fine-tuning
    # ========================
    TEXT_MODEL_UNFREEZE: list = field(default_factory=lambda: [
        'encoder.layer.10',
        'encoder.layer.11',
        'pooler'
    ])

    IMAGE_MODEL_UNFREEZE: list = field(default_factory=lambda: [
        'blocks.6',
        'conv_head',
        'bn2'
    ])
    
    # ========================
    # Архитектура
    # ========================
    TEXT_EMBED_DIM: int = 768
    HIDDEN_DIM: int = 768

    # ========================
    # Оптимизация
    # ========================
    TEXT_LR: float = 5e-5
    IMAGE_LR: float = 5e-3
    PROJECTION_LR: float = 3e-4
    REGRESSOR_LR: float = 3e-4
    
    WEIGHT_DECAY: float = 1e-4
    
    PROJ_DROPOUT: float = 0.2
    FUSION_DROPOUT: float = 0.2
    HEAD_DROPOUT: float = 0.3
    
    EPOCHS: int = 40
    
    # ========================
    # Scheduler
    # ======================== 
    SCHEDULER_FACTOR: float = 0.5
    SCHEDULER_PATIENCE: int = 4
    