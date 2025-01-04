from ultralytics import YOLO


# Build a YOLOv9c model from pretrained weight
model = YOLO("../../train/PTQ_224_416/best_saved_model/best_full_integer_quant.tflite")

# Display model information (optional)
# model.info()

# # Run inference with the YOLOv9c model on the 'bus.jpg' image
results = model.predict("../../video/Video/bentre.mp4", show=True, save=True, imgsz=(224, 416))
