from ultralytics import YOLO

# Load a model
model = YOLO("train/weights/best.pt")
# model = YOLO("train/PTQ_736/best_saved_model/best_int8.tflite")
# model = YOLO("train/PTQ_480/best_saved_model/best_int8.tflite")
# model = YOLO("train/QAT_736/weights/best_saved_model/best_int8.tflite")
# model = YOLO("train/QAT_736/weights/best_saved_model/best_full_integer_quant.tflite")

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category