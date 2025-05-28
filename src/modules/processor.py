import cv2
from . import object_counter
from time import time

FPS_WARMUP_FRAMES = 5


def setup_object_counter(model_names, number_lane, lanes):
    counter = object_counter.ObjectCounter()
    counter.set_args(
        view_img=True,
        reg_pts=lanes,
        classes_names=dict(model_names),
        draw_tracks=True,
        line_thickness=2,
        region_thickness=1,
        track_thickness=1,
        region_lane=number_lane
    )
    return counter


def process_video_frames(cap, model, video_writer, object_count, fps_warmup_frames, img_size):
    total_fps = 0
    frame_count = 0
    min_fps = float("inf")
    max_fps = 0
    inference_time_ms = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video processing completed.")
            break

        frame_start_time = time()

        tracks = model.track(frame, persist=True, show=False, imgsz=img_size, conf=0.1)

        frame, current_fps = object_count.start_counting(frame, tracks, frame_start_time)

        speed = tracks[0].speed
        inference_time_ms += speed['inference']

        if frame_count >= fps_warmup_frames:
            total_fps += current_fps
            min_fps = min(min_fps, current_fps)
            max_fps = max(max_fps, current_fps)

        print(f"FPS: {current_fps:.2f}")

        video_writer.write(frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    average_fps = total_fps / (frame_count - fps_warmup_frames) if frame_count > fps_warmup_frames else 0
    average_inference_time = inference_time_ms / frame_count
    print(f"Average inference time: {average_inference_time:.2f} ms")
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Min FPS: {min_fps:.2f}")
    print(f"Max FPS: {max_fps:.2f}")
    return average_fps, min_fps, max_fps, average_inference_time
