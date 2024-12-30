import cv2

# Hàm callback để xử lý sự kiện click chuột
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Tọa độ của điểm bạn click là: X = {x}, Y = {y}")
        # Vẽ một vòng tròn tại điểm click và hiển thị
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"({x}, {y})"
        cv2.putText(img, text, (x + 10, y - 10), font, 0.8, (255, 0, 0), 2)
        cv2.imshow("Image", img)

# Đọc ảnh từ file
img = cv2.imread('../../video/Frame/bentre_1.jpg')

# Kiểm tra nếu ảnh được đọc thành công
if img is None:
    print("Không thể đọc ảnh!")
else:
    # Hiển thị ảnh và đợi sự kiện click chuột
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)

    # Đợi cho đến khi người dùng nhấn phím để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()
