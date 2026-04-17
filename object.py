import cv2
import numpy as np
from object_detection import ObjectDetector
import sys
import os
import json

# CHỌN CLASS BẠN MUỐN DÒ
TARGET_CLASSES = [0]  # Mặc định là 0: person

print("Loading YOLO11 model...")
od = ObjectDetector(model_path="yolo11m.pt", min_score_thresh=0.35)
print("Model loaded successfully!")

# --- THIẾT LẬP CHƯƠNG 4 (PHÂN ĐOẠN ẢNH) ---
print("Khởi tạo thuật toán Background Subtractor MOG2...")
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# --- MENU CHỌN NGUỒN ---
print("\n" + "="*40)
print("HỆ THỐNG GIÁM SÁT THÔNG MINH (4 CHƯƠNG)")
print("="*40)
print("1. Sử dụng Webcam (Real-time)")
print("2. Sử dụng File Video (people.mkv)")
print("3. Thoát")
print("-"*40)

choice = input("Nhập lựa chọn của bạn (1, 2 hoặc 3): ")

if choice == '1':
    cap = cv2.VideoCapture(0)
    source_label = "Webcam"
elif choice == '2':
    video_path = "D:/NguyenDucThinh/xulyanh/duan/object-detection-opencv/people.mkv"
    cap = cv2.VideoCapture(video_path)
    source_label = f"Video: {video_path}"
else:
    sys.exit()

if not cap.isOpened():
    print(f"\n[LỖI] Không thể mở được {source_label}!")
    sys.exit()

# --- TẢI VÙNG ROI TỰ ĐỘNG TỪ FILE JSON ---
if os.path.exists("roi.json"):
    with open("roi.json", "r") as f:
        data = json.load(f)
        ROI = np.array(data["ROI"], dtype=np.int32)
    print("\n[INFO] Đã tải tọa độ ROI từ roi.json")
else:
    ROI = np.array([ [360, 213], [1121, 204], [1203, 628], [101, 654], [360, 215]], dtype=np.int32)
    print("\n[INFO] Không tìm thấy roi.json, dùng ROI mặc định")

def is_inside_roi(box, roi):
    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    return cv2.pointPolygonTest(roi, (cx, cy), False) > 0

print("\nĐang chạy xử lý... Bấm ESC để thoát, SPACE để tạm dừng.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ====================================================================
    # PHẦN 1: ÁP DỤNG CÁC THUẬT TOÁN XỬ LÝ ẢNH TRUYỀN THỐNG (CHƯƠNG 2, 3, 4)
    # ====================================================================
    
    # 1. CHƯƠNG 2: CÁC THUẬT TOÁN XỬ LÝ ẢNH (Lọc tuyến tính)
    # Áp dụng Gaussian Blur để làm mịn ảnh, khử nhiễu giúp các bước sau chính xác hơn.
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # 2. CHƯƠNG 4: PHÂN ĐOẠN ẢNH (Đường viền động / Trừ nền)
    # Sử dụng MOG2 để tách đối tượng đang di chuyển ra khỏi phông nền.
    fg_mask = backSub.apply(blurred_frame)
    
    # 3. CHƯƠNG 3: PHÁT HIỆN ĐẶC TRƯNG (Phát hiện cạnh)
    # Dùng thuật toán Canny trên ảnh Mask để trích xuất các đường viền đặc trưng.
    edges = cv2.Canny(fg_mask, 50, 150)

    # Hiển thị các bước xử lý ra các cửa sổ riêng biệt để demo cho giáo viên
    # cv2.imshow("Chuong 2: Gaussian Blur", blurred_frame) # (Thường không cần show vì trông giống ảnh gốc)
    cv2.imshow("Chuong 4: Phan doan anh (Foreground Mask)", fg_mask)
    cv2.imshow("Chuong 3: Phat hien dac trung (Canny Edges)", edges)

    # ====================================================================
    # PHẦN 2: NHẬN DẠNG BẰNG AI (CHƯƠNG 5) VÀ XỬ LÝ LOGIC NGHIỆP VỤ
    # ====================================================================
    
    # 4. CHƯƠNG 5: NHẬN DẠNG ẢNH (Nhận dạng và phân loại đối tượng)
    # Đưa frame gốc vào YOLO để đảm bảo AI nhận dạng con người chính xác nhất
    bboxes, class_ids, scores = od.detect(frame, conf=0.35, classes=TARGET_CLASSES)
    
    count = 0
    filtered_boxes = []
    filtered_ids = []
    filtered_scores = []
    
    for box, cid, score in zip(bboxes, class_ids, scores):
        if is_inside_roi(box, ROI):
            count += 1
            filtered_boxes.append(box)
            filtered_ids.append(cid)
            filtered_scores.append(score)
    
    # Vẽ kết quả lên frame
    frame = od.draw_boxes(frame, filtered_boxes, filtered_ids, filtered_scores, line_thickness=2)
    cv2.polylines(frame, [ROI], True, (0, 255, 255), 2)
    
    # Ghi text thông tin
    cv2.putText(frame, f'In ROI Count: {count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Chuong 5: Nhan dang anh & Tong hop (YOLO11)", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord(' '):
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
od.close()