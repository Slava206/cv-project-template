
# Запуск дообучения Faster R-CNN на KITTI

import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.faster_rcnn import FasterRCNNModel
from src.utils.utils import get_device, set_seed
from src.dataset.dataset import TorchvisionDataset, collate_fn

def main():
    # Настройки
    DATA_DIR = 'data'
    BATCH_SIZE = 4
    EPOCHS = 15
    
    set_seed(42)
    device = get_device()
    print(f"Устройство: {device}")
    
    # Загрузка данных
    train_dataset = TorchvisionDataset(DATA_DIR, split='train')
    val_dataset = TorchvisionDataset(DATA_DIR, split='valid')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Создаем модель (адаптированную под KITTI)
    model = FasterRCNNModel(num_classes=7, device=device)
    
    # Дообучаем на KITTI
    model.train(train_loader, val_loader, epochs=EPOCHS, lr=0.001)
    
    # Оцениваем
    model.evaluate(val_loader)
    
    # Сохраняем
    model.save('results/models/faster_rcnn_kitti.pth')
    
    print("\n Готово")

if __name__ == "__main__":
    main()