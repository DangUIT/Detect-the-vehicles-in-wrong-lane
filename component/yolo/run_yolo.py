from ultralytics import YOLO


# Build a YOLOv9c model from pretrained weight
model = YOLO("train/weights/best.pt")

# Display model information (optional)
# model.info()

# # Run inference with the YOLOv9c model on the 'bus.jpg' image
# results = model("Test/Video/test3.mp4", save=True)
