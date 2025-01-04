import argparse
from ultralytics import YOLO
import cv2
import object_counter
from time import time
import os
import json

# Constants
MODEL_PATH = "../train/PTQ_224_416/best_saved_model/best_full_integer_quant.tflite"
FPS_WARMUP_FRAMES = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
TARGET_SIZE = (1280, 720)
IMAGE_SIZE = (224, 416)
LANES_CONFIG_PATH = "../config/lanes_config.json"  # Path to the single JSON configuration file


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a video with the YOLO model.")
    parser.add_argument(
        '--video',
        type=str,
        default='../video/Video/bentre.mp4',  # Default video path if none provided
        help='Path to the input video file'
    )
    return parser.parse_args()


def load_lane_configuration(video_filename):
    # Get the video name without the extension
    video_name = os.path.splitext(os.path.basename(video_filename))[0]

    # Open and read the single JSON configuration file
    if not os.path.exists(LANES_CONFIG_PATH):
        raise FileNotFoundError(f"Lane configuration file not found: {LANES_CONFIG_PATH}")

    with open(LANES_CONFIG_PATH, "r", encoding='utf-8') as file:
        lanes_config = json.load(file)  # Read lane configuration data

    # Check and return the lane configuration for the video
    if video_name not in lanes_config:
        raise ValueError(f"No lane configuration found for video: {video_name}")

    return lanes_config[video_name]


def initialize_model(model_path):
    return YOLO(model_path, task="detect")


def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps


def initialize_video_writer(output_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, fps, TARGET_SIZE)


def get_unique_output_path(video_path, output_dir="../result/video", suffix="_result", extension=".avi"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_base = os.path.join(output_dir, f"{video_name}{suffix}")
    output_path = f"{output_base}{extension}"

    counter = 1
    while os.path.exists(output_path):
        output_path = f"{output_base}_{counter}{extension}"
        counter += 1

    return os.path.normpath(output_path)


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


def process_video_frames(cap, model, video_writer, object_count, fps_warmup_frames):
    total_fps = 0
    frame_count = 0
    min_fps = float("inf")
    max_fps = 0
    start_time = time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video processing completed.")
            break

        current_fps = 1 / (time() - start_time)
        start_time = time()

        if frame_count >= fps_warmup_frames:
            total_fps += current_fps
            min_fps = min(min_fps, current_fps)
            max_fps = max(max_fps, current_fps)

        frame_count += 1
        print(f"FPS: {current_fps:.2f}")

        tracks = model.track(frame, persist=True, show=False, imgsz=IMAGE_SIZE, conf=0.1)
        cv2.putText(frame, f"FPS: {current_fps:.2f}", (27, 61), FONT, 1, (0, 0, 255), 3)
        frame = object_count.start_counting(frame, tracks)

        video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    average_fps = total_fps / (frame_count - fps_warmup_frames) if frame_count > fps_warmup_frames else 0
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Min FPS: {min_fps:.2f}")
    print(f"Max FPS: {max_fps:.2f}")


def main():
    args = parse_arguments()

    # Load lane configuration
    lanes = load_lane_configuration(args.video)

    # Count number lanes
    number_lane = len(lanes)

    # Initialize model
    model = initialize_model(MODEL_PATH)
    print(model.names)

    cap, fps = initialize_video_capture(args.video)

    output_video_path = get_unique_output_path(args.video)
    video_writer = initialize_video_writer(output_video_path, fps)

    object_count = setup_object_counter(model.names, number_lane, lanes)

    process_video_frames(cap, model, video_writer, object_count, FPS_WARMUP_FRAMES)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Video output saved to: {output_video_path}")


if __name__ == "__main__":
    main()
