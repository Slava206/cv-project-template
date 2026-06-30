import sys
import os
from ultralytics import YOLO

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.yolo import YOLOv8Model
from src.utils.utils import get_device, set_seed

def main():
    DATA_YAML = 'data/data.yaml'
    EPOCHS = 25
    BATCH_SIZE = 8
    IMG_SIZE = 640

    device = get_device()
    set_seed(42)

    # Путь к файлу с последней контрольной точкой
    last_weights_path = 'results/logs/yolo/exp/weights/last.pt'

    # Проверяем есть ли уже сохраненная модель
    if os.path.exists(last_weights_path):
        print(f"Найдена контрольная точка: {last_weights_path}")
        print("Возобновляем обучение...")
        model = YOLO(last_weights_path)
        model.train(resume=True)
    else:
        print("Запуск обучение с нуля...")
        model = YOLOv8Model(
            model_name='yolov8n.pt',
            num_classes=7,
            device=device
        )
        model.train(
            data_yaml=DATA_YAML,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            imgsz=IMG_SIZE
        )

if __name__ == "__main__":
    main()