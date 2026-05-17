import random
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def set_seed(seed, return_generator=False):
    """
    Устанавливает seed. При return_generator=True инициализирует 
    generator с заданным seed и возвращает его.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if return_generator:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator


def seed_worker(worker_id):
    """
    Устанавливает seed внутри отдельного Dataloader worker
    для обеспечения воспроизводимости случайных операций при загрузке данных.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def show_images(df, img_path, n, cfg):
    """
    Показывает на графике n случайно отобранных изображений. 
    """
    set_seed(cfg.SEED)
    
    indices = random.sample(range(len(df)), n)
    
    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(10, 10))
    
    for i, idx in enumerate(indices):
        id = df.iloc[idx].name

        image_path = img_path / id / 'rgb.png'
        
        image = Image.open(image_path).convert('RGB')
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(id)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
        

def show_errors(df, img_dir, title=''):
    """
    Строит график топ ошибок модели.
    """
    n = len(df)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4.2 * rows),
        constrained_layout=False
    )

    axes = np.array(axes).reshape(-1)

    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]

        image_path = Path(img_dir) / row['dish_id'] / 'rgb.png'
        image = Image.open(image_path).convert('RGB')

        ax.imshow(image)
        ax.axis('off')

        text = (
            f'{row["dish_id"]}\n'
            f'Real: {row["target"]:.1f} | '
            f'Pred: {row["prediction"]:.1f} | '
            f'Err: {row["abs_error"]:.1f}\n'
            f'{fill(row["ingredients"], width=35)}'
        )

        ax.text(
            0.5,
            -0.06,
            text,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10
        )

    for ax in axes[n:]:
        ax.axis('off')

    fig.subplots_adjust(
        top=0.98,
        bottom=0.05,
        left=0.02,
        right=0.98,
        wspace=0.05,
        hspace=0.35
    )

    plt.show()