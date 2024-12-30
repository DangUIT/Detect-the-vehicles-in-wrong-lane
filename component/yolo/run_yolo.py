from ultralytics import YOLO


# Build a YOLOv9c model from pretrained weight
model = YOLO("../../train/PTQ_416_736/best_int8.tflite")

# Display model information (optional)
# model.info()

# # Run inference with the YOLOv9c model on the 'bus.jpg' image
results = model.predict("../../video/Video/pvd_front.mp4", show= True,save=True, imgsz= (416,736))
