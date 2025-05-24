import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a video with the YOLO model.")
    parser.add_argument(
        '--video',
        type=str,
        default='../video/Video/bentre.mp4',
        help='Path to the input video file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='../train/imgsz_224/best.pt',
        help='Path to the YOLO model file'
    )
    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=('WIDTH', 'HEIGHT'),
        help='Image size as width height (e.g. --image 224 224)'
    )
    return parser.parse_args()