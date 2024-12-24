# from ultralytics import YOLO
#
# # Load the exported TFLite model
# # tflite_model = YOLO("train/weights/best.pt")
#
# # tflite_model = YOLO("../../train/PTQ_416_736/best_full_integer_quant.tflite")
#
# tflite_model = YOLO("../../train/PTQ_416_736/best_int8.tflite")
# # tflite_model = YOLO("../../train/PTQ_384_640/best_saved_model/best_int8.tflite")
#
# # tflite_model.info()
#
# # Run inference
# results = tflite_model.track("../../video/Video/test.mp4", save=True, show=True, imgsz=(416, 736),conf=0.25)


import cv2

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("../../train/PTQ_416_736/best_int8.tflite")

# Open the video file
video_path = "../../video/Video/test.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model.track(frame, imgsz=(416,736))



        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()