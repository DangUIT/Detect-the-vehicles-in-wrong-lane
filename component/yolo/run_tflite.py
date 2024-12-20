from ultralytics import YOLO

# Load the exported TFLite model
# tflite_model = YOLO("train/weights/best.pt")

tflite_model = YOLO("../../train/PTQ_416_736/best_full_integer_quant.tflite")


# tflite_model.info()

# Run inference
results = tflite_model("Test/Video/test3.mp4", save=True, show=True, imgsz=(480, 480))


