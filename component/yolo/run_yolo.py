import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("../../train/imgsz_224/best.pt")

# Open the video file
video_path = "../../video/Video/pvd_front.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
inference_time_ms = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.predict(frame, imgsz=(224, 224))

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        speed = results[0].speed
        inference_time_ms += speed['inference']
        frame_count = frame_count + 1

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
average_inference_time = inference_time_ms / frame_count
print(f"Average inference time: {average_inference_time:.2f} ms")