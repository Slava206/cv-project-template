import sys
import os

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.yolo import YOLOv8Model
from src.utils.utils import get_device, set_seed

def main():
    # Настройки
    DATA_YAML = 'data/data.yaml'
    EPOCHS = 10
    BATCH_SIZE = 8
    IMG_SIZE = 640
    
    # Фиксируем seed для воспроизводимости
    set_seed(42)
    device = get_device()
    
    print(f"Устройство: {device}")
    
    # Создаем модель
    model = YOLOv8Model(
        model_name='yolov8n.pt',
        num_classes=7,
        device=device
    )
    
    # Обучение
    model.train(
        data_yaml=DATA_YAML,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        imgsz=IMG_SIZE
    )
    
    # Оценка
    metrics = model.evaluate(
        data_yaml=DATA_YAML,
        batch_size=BATCH_SIZE,
        imgsz=IMG_SIZE
    )
    
    # Сохранение
    model.save('results/models/yolov8_best.pt')
    
    print("Модель: results/models/yolov8_best.pt")
    print("Графики: results/logs/yolo/exp/")

if __name__ == "__main__":
    main()