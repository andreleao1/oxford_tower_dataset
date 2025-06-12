import torch
import numpy as np
from ultralytics import YOLO

print(torch.__version__)
print(np.__version__)

model = YOLO("trained_by_yolo11n.pt")
model.export(format="onnx", dynamic=True)