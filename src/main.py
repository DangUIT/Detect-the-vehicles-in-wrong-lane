import cv2
from modules.arguments import parse_arguments
from modules.utils import load_lane_configuration, initialize_model, initialize_video_capture, initialize_video_writer
from modules.utils import get_unique_output_path, get_log_file_path
from modules.processor import setup_object_counter, process_video_frames


def main():
    args = parse_arguments()

    # Load lane configuration
    lanes = load_lane_configuration(args.video)
    number_lane = len(lanes)

    # Initialize model
    model = initialize_model(args.model)
    print(model.names)

    # Initialize video capture
    cap, fps = initialize_video_capture(args.video)

    # Read first frame to get the size
    ret, frame = cap.read()
    if not ret:
        print("Can not read first frame")
        return

    frame_height, frame_width = frame.shape[:2]
    frame_size = (frame_width, frame_height)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset cap

    # Initialize output paths
    output_video_path = get_unique_output_path(args.video, args.model, args.size)
    log_path = get_log_file_path(args.video, args.model, args.size)

    # Initialize video writer
    video_writer = initialize_video_writer(output_video_path, fps, frame_size)

    # Setup object counter
    object_count = setup_object_counter(model.names, number_lane, lanes)

    # Process video frames
    average_fps, min_fps, max_fps, avg_infer_time = process_video_frames(
        cap, model, video_writer, object_count, 5, args.size
    )

    # Cleanup
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Save log
    with open(log_path, "w") as log_file:
        log_file.write(f"Average FPS: {average_fps:.2f}\n")
        log_file.write(f"Min FPS: {min_fps:.2f}\n")
        log_file.write(f"Max FPS: {max_fps:.2f}\n")
        log_file.write(f"Average Inference Time: {avg_infer_time:.2f} ms\n")

    print(f"Video output saved to: {output_video_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()