from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from scripts.utils import seed_worker, set_seed


class DishDataset(Dataset):
    """
    Датасет для мультимодального предсказания калорийности блюд.
    """
    def __init__(
        self, 
        df, 
        tokenizer,
        max_length, 
        img_dir,
        mass_mean,
        mass_std,
        transform=None
):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_dir = Path(img_dir)
        self.mass_mean = mass_mean
        self.mass_std = mass_std
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        id = row.name
        path = Path(self.img_dir / id)
        
        image = Image.open(path / 'rgb.png').convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        ingredients = row['ingredients']
        text = 'ingredients: ' + ingredients.replace(';', ', ')
        
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Убираем batch dimension        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Нормализация массы
        mass = (row['total_mass'] - self.mass_mean) / self.mass_std
        mass = torch.tensor(mass, dtype=torch.float32)
        
        target = torch.tensor(row['total_calories'], dtype=torch.float32)
        
        return {
            'dish_id': id,
            'image': image,
            'ingredients': ingredients,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mass': mass,
            'target': target,
        }
        

def get_transforms(cfg):
    """
    Создаёт аугментации для train и validation.
    """
    train_transform = A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.92, 1.08),
            rotate=(-45, 45),
            p=0.5
        ),
        A.Normalize(),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


def create_dataloaders(
    train_df,
    val_df,
    cfg,
    train=True
):
    """
    Создаёт DataLoader для обучения и валидации.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
    
    train_transform, val_transform = get_transforms(cfg)
    
    mass_mean = train_df['total_mass'].mean()
    mass_std = train_df['total_mass'].std()
    
    if train: 
        train_dataset = DishDataset(
            df=train_df,
            tokenizer=tokenizer,
            max_length=cfg.MAX_LENGTH,
            img_dir=cfg.IMAGE_DIR,
            mass_mean=mass_mean,
            mass_std=mass_std,
            transform=train_transform
        )
        
        generator = set_seed(
            seed=cfg.SEED, 
            return_generator=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker if cfg.NUM_WORKERS > 0 else None,
            generator=generator,
            num_workers=cfg.NUM_WORKERS
        )
        
    val_dataset = DishDataset(
        df=val_df,
        tokenizer=tokenizer,
        max_length=cfg.MAX_LENGTH,
        img_dir=cfg.IMAGE_DIR,
        mass_mean=mass_mean,
        mass_std=mass_std,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS
    )
    
    if train:
        return train_loader, val_loader
    
    return val_loader