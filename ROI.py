import cv2
import numpy as np
import json

points = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Xóa điểm cuối
        if points:
            points.pop()

# Load ảnh hoặc frame đầu video
cap = cv2.VideoCapture("people.mkv")
ret, frame = cap.read()
cap.release()

cv2.namedWindow("Draw ROI")
cv2.setMouseCallback("Draw ROI", mouse_callback)

while True:
    temp = frame.copy()

    # Vẽ các điểm
    for p in points:
        cv2.circle(temp, p, 5, (0, 0, 255), -1)

    # Nối thành polygon
    if len(points) > 1:
        cv2.polylines(temp, [np.array(points)], False, (0, 255, 0), 2)

    cv2.imshow("Draw ROI", temp)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    elif key == ord('s'):  # SAVE
        if len(points) >= 3:
            roi = np.array(points, dtype=np.int32).tolist()

            with open("roi.json", "w") as f:
                json.dump({"ROI": roi}, f)

            print("Saved ROI to roi.json")
            break
        else:
            print("Cần ít nhất 3 điểm!")

    elif key == ord('c'):  # CLEAR
        points = []

cv2.destroyAllWindows()