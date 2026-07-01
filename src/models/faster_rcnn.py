
# Faster R-CNN адаптированная под KITTI.


import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from tqdm import tqdm


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
    
    def train(self, train_loader, val_loader, epochs=3, lr=0.001):
        
        # Дообучение на KITTI
        
        print("Дообучение FASTER R-CNN НА KITTI")
        print(f"Эпох: {epochs}")
        print(f"Learning rate: {lr}")
        
        # Оптимизатор SGD
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for images, targets, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
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
            
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        print("\nДообучение завершено")
    
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
        print("\nОценка Faster R-CNN на KITTI...")
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
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
