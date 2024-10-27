from ultralytics import YOLO

# Load model
model = YOLO('D:/Project/train/weights/best.pt')

model.export(format="tflite",
             int8=True,
             imgsz=640,
             data='D:/Project/train/vehicle-detection-9/data.yaml',
             optimize=True
             )