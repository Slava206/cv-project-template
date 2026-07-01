from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Tuple, Dict
import os


class YOLOv8Model:    
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        num_classes: int = 7,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.num_classes = num_classes
        self.model_name = model_name
        
        print(f"Загрузка YOLOv8: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)
        
        print(f"YOLOv8 загружена на {device}")
        print(f"Классов: {num_classes}")
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 15,
        batch_size: int = 4,
        imgsz: int = 320,
        lr: float = 0.001,
        optimizer: str = 'AdamW',
        weight_decay: float = 0.0005,
        momentum: float = 0.937,
        patience: int = 5
    ):
        print("НАЧАЛО ОБУЧЕНИЯ YOLOv8")
        print(f"{'='*50}")
        print(f"Эпох: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Размер изображений: {imgsz}")
        print(f"Learning rate: {lr}")
        
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'lr0': lr,
            'optimizer': optimizer,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'patience': patience,
            'project': 'results/logs/yolo',
            'name': 'exp',
            'exist_ok': True,
            'pretrained': True,
            'amp': True,
            'plots': True,
            'verbose': True,
        }
        
        results = self.model.train(**train_args)
        
        print(f"\nОбучение завершено")
        print(f"Результаты: results/logs/yolo/exp/")
        
        return results
    
    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        results = self.model(image, conf=confidence_threshold, iou=iou_threshold)
        
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            return boxes, scores, classes
        
        return np.array([]), np.array([]), np.array([])
    
    def evaluate(
        self,
        data_yaml: str,
        batch_size: int = 8,
        imgsz: int = 640,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict:
        print(f"\nОценка YOLOv8...")
        
        metrics = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=imgsz,
            conf=confidence_threshold,
            iou=iou_threshold
        )
        
        result = {
            'mAP50': metrics.box.map50,
            'mAP50_95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
            'f1': metrics.box.f1
        }
        
        print(f"   mAP@0.5: {result['mAP50']:.4f}")
        print(f"   mAP@0.5:0.95: {result['mAP50_95']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall: {result['recall']:.4f}")
        print(f"   F1: {float(result['f1']):.4f}")
        
        return result
    
    def save(self, path: str):
        self.model.save(path)
        print(f"Модель сохранена в {path}")
    
    def load(self, path: str):
        self.model = YOLO(path)
        print(f"Модель загружена из {path}")