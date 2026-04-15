import cv2
from object_detection import ObjectDetector

# Khởi tạo detector
print("Loading YOLO11 model...")
od = ObjectDetector(model_path="yolo11m.pt", min_score_thresh=0.35)
print("Model loaded successfully!")

# IN RA DANH SÁCH CÁC CLASS (80 đối tượng)
print("\n=== DANH SÁCH CÁC ĐỐI TƯỢNG NHẬN DIỆN ===\n")
for idx, class_name in od.classes.items():
    print(f"{idx}: {class_name}")
print(f"\nTổng cộng: {len(od.classes)} đối tượng")

# Load video
cap = cv2.VideoCapture("demo.mkv")

# Nếu không mở được video thì dùng webcam
if not cap.isOpened():
    print("Cannot open video file, using webcam...")
    cap = cv2.VideoCapture(0)

print("\nPress ESC to exit, SPACE to pause")

# CHỌN CLASS BẠN MUỐN DÒ
TARGET_CLASSES = [2]  # Thay đổi theo nhu cầu


while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame")
        break
    
    # Detect objects - CHỈ LẤY CÁC CLASS ĐÃ CHỌN
    bboxes, class_ids, scores = od.detect(frame, conf=0.35, classes=TARGET_CLASSES)
    
    # Draw bounding boxes
    frame = od.draw_boxes(frame, bboxes, class_ids, scores, line_thickness=2)
    
    # Show info
    cv2.putText(frame, f'Objects: {len(bboxes)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'ESC to exit', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Show frame
    cv2.imshow("YOLO11 Object Detection", frame)
    
    # Handle keys
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE to pause
        cv2.waitKey(0)

# Release resources
cap.release()
cv2.destroyAllWindows()
od.close()
print("Program ended!")