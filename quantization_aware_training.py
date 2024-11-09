import torch
import torch.quantization
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov9c.pt')

# Set the model to training mode
model.model.train()

# Fuse model layers (necessary for QAT in PyTorch)
model.model.fuse()

# Prepare the model for quantization-aware training (QAT)
model.model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
torch.quantization.prepare_qat(model.model, inplace=True)

# Train the model as usual
model.train(data='D:/Project/train/vehicle-detection-9/data.yaml', epochs=5, imgsz=640, batch=4)