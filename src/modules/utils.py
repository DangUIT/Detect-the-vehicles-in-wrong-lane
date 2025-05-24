import os
import json
import cv2
from ultralytics import YOLO
from datetime import datetime

LANES_CONFIG_PATH = "../config/lanes_config.json"


def load_lane_configuration(video_filename):
    video_name = os.path.splitext(os.path.basename(video_filename))[0]

    if not os.path.exists(LANES_CONFIG_PATH):
        raise FileNotFoundError(f"Lane configuration file not found: {LANES_CONFIG_PATH}")

    with open(LANES_CONFIG_PATH, "r", encoding='utf-8') as file:
        lanes_config = json.load(file)

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


def initialize_video_writer(output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def get_model_name_from_path(model_path):
    model_path = os.path.normpath(model_path)
    base = os.path.basename(model_path)
    name, ext = os.path.splitext(base)
    return name if ext else base


def get_unique_path(video_path, model_path, size, output_dir, extension):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    model_name = get_model_name_from_path(model_path)
    width, height = size

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{video_name}_{model_name}_{width}x{height}_{timestamp}{extension}"
    output_path = os.path.join(output_dir, filename)

    return os.path.normpath(output_path)


def get_unique_output_path(video_path, model_path, size, output_dir="../result/video", extension=".avi"):
    return get_unique_path(video_path, model_path, size, output_dir, extension)


def get_log_file_path(video_path, model_path, size, output_dir="../result/benchmark/", extension=".txt"):
    return get_unique_path(video_path, model_path, size, output_dir, extension)