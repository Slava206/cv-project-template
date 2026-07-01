# Faster R-CNN адаптированная под KITTI.

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import json


class FasterRCNNModel:
    """
    Faster R-CNN для KITTI (7 классов).
    Использует предобученный бэкбон ResNet-50.
    """
 
    def __init__(self, num_classes=7, device='cpu'):
        self.device = device
        self.num_classes = num_classes + 1  # +1 для фона
        
        # Загружаем предобученную модель (на COCO)
        print("Загрузка предобученной модели")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        
        # ЗАМЕНЯЕМ ГОЛОВКУ ДЛЯ 7 КЛАССОВ KITTI
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        self.model.to(device)
        print(f"Faster R-CNN адаптирована для {num_classes} классов KITTI")
        print(f"Устройство: {device}")
    
    def train(self, train_loader, val_loader, epochs=15, lr=0.001, save_best=True, save_dir='checkpoints'):
        """
        Дообучение Faster R-CNN на KITTI
        """
        # Создаем папку для чекпоинтов
        os.makedirs(save_dir, exist_ok=True)
        
        print("Дообучение FASTER R-CNN НА KITTI")
        print(f"Эпох: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Чекпоинты сохраняются в: {save_dir}")
        
        # Оптимизатор
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        self.model.train()
        best_loss = float('inf')
        history = {'epochs': [], 'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Прогресс-бар
            loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for images, targets, _ in loop:
                # Перенос на GPU
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                total_loss += losses.item()
                
                # Обновляем прогресс-бар
                loop.set_postfix(loss=losses.item())
            
            scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
            
            # Валидация
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                print(f"  Val Loss: {val_loss:.4f}")
            
            # Сохраняем историю
            history['epochs'].append(epoch + 1)
            history['train_loss'].append(avg_loss)
            history['val_loss'].append(val_loss if val_loss is not None else 0)
            
            # Сохраняем чекпоинт каждой эпохи
            checkpoint_path = f'{save_dir}/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  Чекпоинт сохранен: {checkpoint_path}")
            
            # Сохраняем лучшую модель
            if save_best and avg_loss < best_loss:
                best_loss = avg_loss
                best_path = f'{save_dir}/best_model_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'val_loss': val_loss,
                }, best_path)
                print(f"  Лучшая модель сохранена (loss: {best_loss:.4f})")
            
            # Сохраняем историю
            with open(f'{save_dir}/training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
        
        print(f"\n Дообучение завершено! Лучший loss: {best_loss:.4f}")
        return best_loss
    
    def _validate(self, val_loader):
        """
        Валидация модели на валидационном наборе данных.
        """
        # Сохраняем текущий режим
        was_training = self.model.training
        
        # Переключаем в режим train для вычисления loss
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # В режиме train модель возвращает словарь с потерями
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                num_batches += 1
        
        # Восстанавливаем исходный режим
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        
        if num_batches == 0:
            return 0.0
        
        return total_loss / num_batches
    
    def predict(self, image, threshold=0.5):
        # Предсказание на одном изображении
        self.model.eval()
        
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            predictions = self.model(image)[0]
        
        # Фильтруем по уверенности
        mask = predictions['scores'] >= threshold
        boxes = predictions['boxes'][mask].cpu().numpy()
        scores = predictions['scores'][mask].cpu().numpy()
        labels = predictions['labels'][mask].cpu().numpy() - 1  # -1 для KITTI (0-6)
        
        return boxes, scores, labels
    
    def evaluate(self, val_loader):
        print("\n Оценка Faster R-CNN на KITTI...")
        self.model.eval()
        
        total_boxes = 0
        total_gt = 0
        
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = [img.to(self.device) for img in images]
                predictions = self.model(images)
                
                for pred, target in zip(predictions, targets):
                    total_boxes += len(pred['boxes'])
                    total_gt += len(target['boxes'])
        
        print(f"  Предсказано боксов: {total_boxes}")
        print(f"  GT объектов: {total_gt}")
        print(f"  (Для точных метрик нужен COCOeval)")
        
        return {
            'total_predictions': total_boxes,
            'total_ground_truth': total_gt,
            'note': 'После дообучения на KITTI'
        }
    
    def save(self, path):
        """Сохраняет модель"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f" Модель сохранена: {path}")
    
    def load(self, path):
        """Загружает модель"""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f" Модель загружена: {path}")
        else:
            print(f" Файл не найден: {path}")