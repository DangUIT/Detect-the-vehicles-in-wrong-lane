import cv2
import os

def extract_frames(video_path, output_folder):
    # Lấy tên video mà không có phần mở rộng
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Mở video
    cap = cv2.VideoCapture(video_path)

    # Kiểm tra xem video có mở thành công không
    if not cap.isOpened():
        print("Không thể mở video!")
        return

    # Đếm số frame đã lấy
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()  # Đọc một frame
        if not ret:
            break  # Nếu không có frame nào thì thoát
        frame = cv2.resize(frame, (1280, 720))
        # Mỗi lần cắt 1 frame
        if frame_count % (int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 5)) == 0 and saved_count < 1:
            filename = f"{output_folder}/{video_name}_{saved_count + 1}.jpg"
            cv2.imwrite(filename, frame)  # Lưu ảnh
            saved_count += 1

        frame_count += 1

        if saved_count == 1:  # Đã lấy đủ 1 ảnh
            break

    cap.release()  # Đóng video



extract_frames('../../video/Video/bentre.mp4', '../../video/Frame')
extract_frames('../../video/Video/pvd_behind.mp4', '../../video/Frame')
extract_frames('../../video/Video/pvd_front.mp4', '../../video/Frame')