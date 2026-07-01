# Запуск дообучения Faster R-CNN на KITTI (Colab версия)

import sys
import os
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.faster_rcnn import FasterRCNNModel
from src.utils.utils import get_device, set_seed
from src.dataset.dataset import TorchvisionDataset, collate_fn

def main():
    # Настройки
    DATA_DIR = 'data'
    BATCH_SIZE = 4
    EPOCHS = 15
    LR = 0.001
    
    # Установка seed для воспроизводимости
    set_seed(42)
    
    # Определение устройства
    device = get_device()
    print(f" Устройство: {device}")
    
    if device.type == 'cuda':
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} ГБ")
        torch.backends.cudnn.benchmark = True
    
    # Загрузка данных
    print("\n Загрузка данных...")
    train_dataset = TorchvisionDataset(DATA_DIR, split='train')
    val_dataset = TorchvisionDataset(DATA_DIR, split='valid')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f" Train: {len(train_dataset)} изображений")
    print(f" Val: {len(val_dataset)} изображений")
    
    # Создаем модель
    print("\n Инициализация модели...")
    model = FasterRCNNModel(num_classes=7, device=device)
    
    # Проверка модели на GPU
    print(f" Модель на {next(model.model.parameters()).device}")
    
    # Дообучаем на KITTI
    print("\n Начинаем обучение...")
    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        save_best=True,
        save_dir='checkpoints'  # Сохраняем локально
    )
    
    # Оцениваем
    print("\n Оценка модели...")
    metrics = model.evaluate(val_loader)
    print(f" Метрики: {metrics}")
    
    # Сохраняем финальную модель ЛОКАЛЬНО
    os.makedirs('results/models', exist_ok=True)
    model_path = 'results/models/faster_rcnn_kitti_final.pth'
    model.save(model_path)
    print(f" Модель сохранена локально: {model_path}")
    
    # СОХРАНЯЕМ НА GOOGLE DRIVE (для Colab)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        DRIVE_DIR = '/content/drive/MyDrive/cv_project'
        os.makedirs(DRIVE_DIR, exist_ok=True)
        os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
        
        # Копируем финальную модель
        import shutil
        drive_model_path = f'{DRIVE_DIR}/faster_rcnn_kitti_final.pth'
        shutil.copy2(model_path, drive_model_path)
        print(f" Модель скопирована на Google Drive: {drive_model_path}")
        
        # Копируем чекпоинты
        if os.path.exists('checkpoints'):
            for file in os.listdir('checkpoints'):
                if file.endswith('.pth'):
                    src = f'checkpoints/{file}'
                    dst = f'{DRIVE_DIR}/checkpoints/{file}'
                    shutil.copy2(src, dst)
            print(f" Чекпоинты скопированы на Google Drive: {DRIVE_DIR}/checkpoints/")
        
        # Сохраняем метрики
        import json
        metrics_path = f'{DRIVE_DIR}/metrics.json'
        metrics['epochs'] = EPOCHS
        metrics['batch_size'] = BATCH_SIZE
        metrics['learning_rate'] = LR
        metrics['device'] = str(device)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f" Метрики сохранены: {metrics_path}")
        
        print("\n Все результаты сохранены на Google Drive!")
        print(f" Путь: {DRIVE_DIR}")
        
    except Exception as e:
        print(f" Не удалось сохранить на Google Drive: {e}")
        print(" Модель сохранена локально в папке results/models/")
    
    print("\n Обучение завершено успешно!")

if __name__ == "__main__":
    main()