from ultralytics import YOLO

# Load model
model = YOLO('D:/Project/train/weights/best_qat.pt')

model.export(format="tflite",
             int8=True,
             imgsz=(480, 480),
             data='D:/Project/train/vehicle-detection-9/data.yaml',
             optimize=True
             )