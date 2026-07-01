# Загрузка данных для моделей torchvision

import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class TorchvisionDataset(Dataset):
    # Датасет для Faster R-CNN, SSD, DETR
    
    KITTI_CLASSES = [
        'Car', 'Van', 'Truck', 'Pedestrian', 
        'Person_sitting', 'Cyclist', 'Tram'
    ]
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        
        self.image_dir = os.path.join(data_dir, split, 'images')
        self.label_dir = os.path.join(data_dir, split, 'labels')
        
        # Проверка существования папок
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Папка {self.image_dir} не найдена!")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Папка {self.label_dir} не найдена!")
        
        self.images = [f for f in os.listdir(self.image_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.images.sort()
        
        print(f"Загружено {len(self.images)} изображений для {split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Загрузка изображения
        image = cv2.imread(img_path)
        if image is None:
            print(f"Не удалось загрузить {img_path}")
            return self.__getitem__((idx + 1) % len(self.images))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Загрузка аннотаций
        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx = float(parts[1]) * width
                        cy = float(parts[2]) * height
                        w = float(parts[3]) * width
                        h = float(parts[4]) * height
                        
                        # Проверка валидности
                        if w > 0 and h > 0:
                            x1 = cx - w/2
                            y1 = cy - h/2
                            x2 = cx + w/2
                            y2 = cy + h/2
                            
                            # Клиппинг границ
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(width, x2)
                            y2 = min(height, y2)
                            
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id + 1)  # +1 для фона в Faster R-CNN
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        # Нормализация [0, 1]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Вычисление площади
        if len(boxes) > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros(0)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        return image, target, img_name


def collate_fn(batch):
    # Объединяем батч
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    names = [item[2] for item in batch]
    return images, targets, names