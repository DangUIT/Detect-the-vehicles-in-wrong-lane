from ultralytics import YOLO
import cv2
import object_counter
from time import time

model = YOLO("../train/weight/best.pt")
# model = YOLO("../train/PTQ_416_736/best_int8.tflite", task="detect")
# model = YOLO("../train/PTQ_416_736/best_full_integer_quant.tflite", task="detect")


image_size = (360, 640)

cap = cv2.VideoCapture("../video/Video/test.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

lanes = [[(171, 434), (459, 100), (623, 100), (862, 434)],
         [(171, 434), (459, 100), (521, 100), (426, 434)],
         [(426, 434), (521, 100), (578, 100), (666, 434)],
         [(666, 434), (578, 100), (623, 100), (862, 434)]]


classes = [0, 1, 2, 3]

print(model.names)
# Video writer
video_writer = cv2.VideoWriter("../result/Video/result.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# Init Object Counter
counter_lane_all = object_counter.ObjectCounter()
counter_lane_all.set_args(view_img=True,
                          reg_pts=lanes,
                          classes_names=model.names,
                          draw_tracks=True,
                          line_thickness=2,
                          region_thickness=1)
for i in range(1, 4):
    f = open("../result/Save/Data/lane%d.txt" % i, "w")
    f.write("Wong lane %d\n" % i)
    f.close()

# Variables for calculating average, minimum and maximum FPS
total_fps = 0
frame_count = 0
min_fps = float("inf")
max_fps = 0
warmup_frames = 5

loop_time = time()

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculating for current FPS
    current_fps = 1 / (time() - loop_time)
    loop_time = time()

    # Skip first few frames to stabilize FPS
    if frame_count >= warmup_frames:
        total_fps += current_fps
        min_fps = min(min_fps, current_fps)
        max_fps = max(max_fps, current_fps)

    frame_count += 1
    print(f"FPS: {current_fps:.2f}")
    cv2.putText(im0, '1', (298, 430), font, 1, (0, 255, 255), 1)
    cv2.putText(im0, '2', (546, 430), font, 1, (0, 255, 255), 1)
    cv2.putText(im0, '3', (764, 430), font, 1, (0, 255, 255), 1)
    tracks = model.track(im0, persist=True, show=False, imgsz=image_size, conf=0.25)
    im0 = counter_lane_all.start_counting(im0, tracks)
    video_writer.write(im0)
    cv2.putText(im0, f"FPS: {current_fps:.2f}", (27, 61), font, 1, (0, 0, 255), 3)



cap.release()
video_writer.release()
cv2.destroyAllWindows()

average_fps = total_fps / (frame_count - warmup_frames) if frame_count > warmup_frames else 0
print(f"Average FPS: {average_fps:.2f}")
print(f"Min FPS: {min_fps:.2f}")
print(f"Max FPS: {max_fps:.2f}")
