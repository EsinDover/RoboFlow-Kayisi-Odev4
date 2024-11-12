# Gerekli kütüphaneleri yükleyin
# pip install torch
# pip install ultralytics
# pip install roboflow

import torch
from ultralytics import YOLO
from roboflow import Roboflow

# CUDA ve GPU kullanılabilirliğini kontrol edin
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 modelini başlatın
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli

# Roboflow API anahtarı ile kayısı projesini indirin
rf = Roboflow(api_key="O9Dg0SYhNuW8uSbNg2VJ")
project = rf.workspace("esin").project("kayisi-b8ttg")
version = project.version(2)
dataset = version.download("yolov8")

# Modeli eğit
if __name__ == "__main__":
    model.train(data=f"{dataset.location}/data.yaml", epochs=20)
