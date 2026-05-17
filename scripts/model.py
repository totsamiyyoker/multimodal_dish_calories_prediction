from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn  
from torch.optim import AdamW  
from torchmetrics.regression import MeanAbsoluteError
from transformers import AutoModel

from scripts.dataset import create_dataloaders
from scripts.utils import set_seed


class MultimodalModel(nn.Module):
    """
    Мультимодальная модель для предсказания калорийности блюда.
    
    Модель использует три источника данных:
    - текстовое описание ингредиентов (BERT),
    - изображение блюда (EfficientNet),
    - числовой признак массы блюда.
    
    Признаки из каждой модальности приводятся к общему скрытому
    пространству, объединяются и передаются в регрессионную голову
    для предсказания общей калорийности.
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Текстовый энкодер для обработки списка ингредиентов
        self.text_encoder = AutoModel.from_pretrained(cfg.TEXT_MODEL_NAME)
        
        # Визуальный энкодер для извлечения признаков изображения
        self.image_encoder = timm.create_model(
            cfg.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        
        # Проекция текстовых эмбеддингов в общее пространство признаков
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.TEXT_EMBED_DIM, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.PROJ_DROPOUT)
        )
        
        # Проекция визуальных признаков в общее пространство признаков
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_encoder.num_features, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.PROJ_DROPOUT)
        )
        
        # Проекция числового признака массы
        self.mass_proj = nn.Linear(1, cfg.HIDDEN_DIM)
        
        # Dropout для конкатенированных признаков
        self.fusion_dropout = nn.Dropout(cfg.FUSION_DROPOUT)
        
        # Регрессионная голова для финального предсказания калорийности
        self.regressor = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 3, cfg.HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(cfg.HEAD_DROPOUT),
            nn.Linear(cfg.HIDDEN_DIM * 2, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.HEAD_DROPOUT),
            nn.Linear(cfg.HIDDEN_DIM, 1)
        )

    def _mean_pooling(self, last_hidden_state, attention_mask):
        """
        Выполняет mean pooling по токенам с учётом attention mask.
        
        Усредняет только реальные токены, игнорируя padding.
        Используется для получения одного текстового эмбеддинга
        фиксированной размерности из последовательности токенов.
        """
        mask = attention_mask.unsqueeze(-1).float()
        
        # Сумма эмбеддингов только по реальным токенам
        summed = (last_hidden_state * mask).sum(dim=1)
        
        # Количество реальных токенов
        counts = mask.sum(dim=1).clamp(min=1e-9)
        
        return summed / counts
    
    def forward(
        self,
        input_ids,
        attention_mask,
        image,
        mass
    ):
        """
        Выполняет прямой проход модели.
        """
        
        # Получение текстовых признаков
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Агрегация токеновых представлений в один эмбеддинг
        text_emb = self._mean_pooling(
            text_output.last_hidden_state,
            attention_mask
        )
        
        # Получение визуальных признаков
        image_emb = self.image_encoder(image)
        
        # Приведение признаков к общей размерности
        text_emb = self.text_proj(text_emb)
        image_emb = self.image_proj(image_emb)
        
        mass = mass.unsqueeze(1)
        mass_emb = self.mass_proj(mass)
        
        # Объединение всех модальностей
        features = torch.cat([text_emb, image_emb, mass_emb], dim=1)
        
        # Dropout
        features = self.fusion_dropout(features)
        
        # Финальное предсказание
        output = self.regressor(features)
        
        return output.squeeze(1)
    
            
def set_requires_grad(
    model, 
    unfreeze_patterns=None, 
    verbose=False,
    model_name=''
):
    """
    Выполняет заморозку слоёв модели по переданному паттерну.
    """
    if unfreeze_patterns is None:
        unfreeze_patterns = []
        
    if isinstance(unfreeze_patterns, str):
        unfreeze_patterns = unfreeze_patterns.split('|') if unfreeze_patterns else []
    
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        is_trainable = any(
            name.startswith(pattern)
            for pattern in unfreeze_patterns
        )
        
        param.requires_grad = is_trainable
        
        total_params += param.numel()
        
        if is_trainable:
            trainable_params += param.numel()
        
    if verbose:
        header = f'[{model_name}]' if model_name else '[Model]'
        
        print(f'{header}')
        print(f'Trainable params: {trainable_params:,} / {total_params:,}')
        print(f'Trainable ratio: {trainable_params / total_params:.2%}')
        print(f'Unfreeze_patterns: {unfreeze_patterns}\n')


def get_lrs(optimizer):
    """
    Возвращает словарь с группами параметров оптимизатора.
    """
    return {
        group.get('name', f'group_{i}'): group['lr']
        for i, group in enumerate(optimizer.param_groups)
    }


def train(cfg):
    """
    Выполняет обучение модели.
    """
    device = cfg.DEVICE
    
    set_seed(cfg.SEED)
    
    model = MultimodalModel(cfg)
    model.to(device)
    
    # Разморозка слоёв
    set_requires_grad(
        model.text_encoder, 
        unfreeze_patterns=cfg.TEXT_MODEL_UNFREEZE,
        verbose=True,
        model_name='Text encoder (BERT)'
    )
    
    set_requires_grad(
        model.image_encoder, 
        unfreeze_patterns=cfg.IMAGE_MODEL_UNFREEZE,
        verbose=True,
        model_name='Image encoder (EfficientNet)'
    )
    
    # Оптимизатор
    optimizer = AdamW([
        {
            'params': [p for p in model.text_encoder.parameters() if p.requires_grad], 
            'lr': cfg.TEXT_LR, 
            'name': 'text_encoder'
        },
        {
            'params': [p for p in model.image_encoder.parameters() if p.requires_grad], 
            'lr': cfg.IMAGE_LR, 
            'name': 'image_encoder'
        },
        {
            'params': model.text_proj.parameters(), 
            'lr': cfg.PROJECTION_LR, 
            'name': 'text_proj'
        },
        {
            'params': model.image_proj.parameters(), 
            'lr': cfg.PROJECTION_LR, 
            'name': 'image_proj'
        },
        {
            'params': model.mass_proj.parameters(), 
            'lr': cfg.PROJECTION_LR, 
            'name': 'mass_proj'
        },
        {
            'params': model.regressor.parameters(), 
            'lr': cfg.REGRESSOR_LR, 
            'name': 'regressor'
        }
    ], weight_decay=cfg.WEIGHT_DECAY)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE
    )
    
    # Loss
    criterion = nn.SmoothL1Loss(beta=30.0)
    
    # Загрузка данных
    df = pd.read_csv(cfg.PROCESSED_DATA_PATH, index_col='dish_id')

    train_loader, val_loader = create_dataloaders(
        train_df=df[df['split'] == 'train'],
        val_df=df[df['split'] == 'test'],
        cfg=cfg
    )
    
    print('Обучение модели...\n')
    
    # Цикл обучения
    best_mae = float('inf')
    best_epoch = 0
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            
            targets = batch['target'].to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss/len(train_loader)
        # Валидация
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        
        print(
            f'Epoch {epoch+1}/{cfg.EPOCHS} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'Val mae: {val_mae:.4f}'
        )
        
        # Cтарое значение lr
        old_lrs = get_lrs(optimizer)
        # Делаем шаг scheduler
        scheduler.step(val_mae)
        # Новое значение lr
        new_lrs = get_lrs(optimizer)

        # Если lr поменялся, делаем отладочный вывод для всех групп
        if new_lrs != old_lrs and (epoch + 1) < cfg.EPOCHS:
            print('Plateu reached, LR reduced:')
            
            for name in old_lrs:
                if old_lrs[name] != new_lrs[name]:
                    print(f'    {name}: {old_lrs[name]:.2e} -> {new_lrs[name]:.2e}')
                    
                    
        # Сохранение лучшей модели
        if val_mae < best_mae:
            best_epoch = epoch
            best_mae = val_mae
            Path('models').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.SAVE_MODEL_PATH)
    
    # Загрузка лучшего состояния модели
    print('\nЗагружаем лучшее состояние модели:')
    print(f'Лучшая эпоха: {best_epoch + 1}, значение MAE: {best_mae:.2f}')
    model.load_state_dict(torch.load(cfg.SAVE_MODEL_PATH, map_location=device))
    return model
    

def validate(model, val_loader, criterion, device):
    """
    Выполняет валидацию модели.
    """
    model.eval()
    
    mae_metric = MeanAbsoluteError().to(device)
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            targets = batch['target'].to(device)
            outputs = model(**inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            mae_metric.update(outputs, targets)
        
    avg_loss = total_loss / len(val_loader)    
    mae = mae_metric.compute().item()
    
    return avg_loss, mae


def predict(model, cfg):
    """
    Выполняет предсказания, возвращает датафреймы 
    со всеми предсказаниями, топ-5 максимальных ошибок и
    топ-5 минимальных ошибок.
    """
    device = cfg.DEVICE
    # Загрузка данных
    df = pd.read_csv(cfg.PROCESSED_DATA_PATH, index_col='dish_id')
    
    loader = create_dataloaders(
        train_df=df[df['split'] == 'train'],
        val_df=df[df['split'] == 'test'],
        cfg=cfg,
        train=False
    )
    
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device)
            }
            
            outputs = model(**inputs)
            
            for i in range(len(outputs)):
                predictions.append({
                    'dish_id': batch['dish_id'][i],
                    'ingredients': batch['ingredients'][i],
                    'target': batch['target'][i].item(),
                    'prediction': outputs[i].item()
                })
                
    pred_df = pd.DataFrame(predictions)
    pred_df['abs_error'] = abs(
        pred_df['prediction'] - pred_df['target']
    )
    
    top5_largest = pred_df.nlargest(5, 'abs_error')
    top5_smallest = pred_df.nsmallest(5, 'abs_error')
    
    return pred_df, top5_largest, top5_smallest