import cv2
import numpy as np
from object_detection import ObjectDetector
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

# === CẤU HÌNH ===
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

print("Loading YOLO11 model...")
od = ObjectDetector(model_path="fruit.pt", min_score_thresh=0.35)
print("Model loaded!\n")

# Mở video
cap = cv2.VideoCapture("demo.mkv")

# Load ROI
if os.path.exists("roi.json"):
    with open("roi.json", "r") as f:
        data = json.load(f)
        ROI = np.array(data["ROI"], dtype=np.int32)
else:
    ROI = np.array([[0, 278], [1277, 145], [1266, 718], [6, 713], [1, 280]], dtype=np.int32)

# Tính tỉ lệ
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale_x = DESIRED_WIDTH / original_width
scale_y = DESIRED_HEIGHT / original_height
ROI_display = (ROI * [scale_x, scale_y]).astype(np.int32)

# === THỐNG KÊ ===
stats = {
    'total_frames': 0,
    'total_detections': 0,
    'detections_in_roi': 0,
    'confidence_scores': []
}

print("\nTHI NGHIEM DANH GIA MODEL...")
print("="*60)

frame_num = 0
all_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_num += 1
    stats['total_frames'] += 1
    
    # Resize để hiển thị
    frame_display = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
    ROI_display = (ROI * [scale_x, scale_y]).astype(np.int32)
    
    # Detect objects
    bboxes, class_ids, scores = od.detect(frame, conf=0.35)
    
    # Scale boxes
    bboxes_display = []
    for box in bboxes:
        x, y, w, h = box
        bboxes_display.append([
            int(x * scale_x),
            int(y * scale_y),
            int(w * scale_x),
            int(h * scale_y)
        ])
    
    # Lọc trong ROI
    detections_in_roi = []
    scores_in_roi = []
    for box_disp, score in zip(bboxes_display, scores):
        cx = box_disp[0] + box_disp[2]//2
        cy = box_disp[1] + box_disp[3]//2
        if cv2.pointPolygonTest(ROI_display, (cx, cy), False) > 0:
            detections_in_roi.append(box_disp)
            scores_in_roi.append(score)
            all_detections.append({'frame': frame_num, 'box': box_disp, 'score': score})
    
    # Cập nhật thống kê
    stats['total_detections'] += len(detections_in_roi)
    stats['detections_in_roi'] += len(detections_in_roi)
    stats['confidence_scores'].extend(scores_in_roi)
    
    # === VẼ KẾT QUẢ TRỰC QUAN ===
    # Vẽ detection với màu theo confidence
    for box_disp, score in zip(detections_in_roi, scores_in_roi):
        if score > 0.7:
            color = (0, 255, 0)  # Xanh - tốt
        elif score > 0.5:
            color = (0, 255, 255)  # Vàng - trung bình
        else:
            color = (0, 0, 255)  # Đỏ - yếu
        
        cv2.rectangle(frame_display, (box_disp[0], box_disp[1]), 
                     (box_disp[0]+box_disp[2], box_disp[1]+box_disp[3]), color, 2)
        cv2.putText(frame_display, f"{score:.2f}", 
                   (box_disp[0], box_disp[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Vẽ ROI
    cv2.polylines(frame_display, [ROI_display], True, (255, 255, 0), 2)
    
    # Hiển thị thông tin
    cv2.putText(frame_display, f"Frame: {frame_num}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_display, f"Detections: {len(detections_in_roi)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Chú thích
    cv2.putText(frame_display, "Green: Conf>0.7 | Yellow: 0.5-0.7 | Red: <0.5", 
               (10, DESIRED_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imshow("Thi nghiem - Xanh(VIP) | Vang(TB) | Do(Yeu)", frame_display)
    
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
od.close()

# === TÍNH TOÁN CHỈ SỐ ===
print("\n" + "="*60)
print("KET QUA THI NGHIEM DANH GIA MODEL")
print("="*60)

# 1. Thống kê cơ bản
print(f"\n1. THONG KE CO BAN:")
print(f"   - Tong so frames xu ly: {stats['total_frames']}")
print(f"   - Tong so lan phat hien trong ROI: {stats['detections_in_roi']}")
print(f"   - Trung binh detections/frame: {stats['detections_in_roi']/stats['total_frames']:.2f}")

# 2. Confidence Score
if stats['confidence_scores']:
    print(f"\n2. CONFIDENCE SCORE:")
    print(f"   - Trung binh: {np.mean(stats['confidence_scores']):.3f}")
    print(f"   - Trung vi: {np.median(stats['confidence_scores']):.3f}")
    print(f"   - Do lech chuan: {np.std(stats['confidence_scores']):.3f}")
    print(f"   - Min: {np.min(stats['confidence_scores']):.3f}")
    print(f"   - Max: {np.max(stats['confidence_scores']):.3f}")

# 3. Precision ước lượng
if stats['confidence_scores']:
    print(f"\n3. CHI SO PRECISION & RECALL:")
    print(f"   - Chua co ground truth de tinh Precision/Recall chinh xac")
    est_precision = sum(1 for s in stats['confidence_scores'] if s > 0.5) / len(stats['confidence_scores'])
    print(f"   - Precision uoc luong (theo confidence > 0.5): {est_precision*100:.1f}%")

# 4. Phân bố confidence
if stats['confidence_scores']:
    print(f"\n4. PHAN BO CONFIDENCE THEO NGUONG:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        count = sum(1 for s in stats['confidence_scores'] if s > thresh)
        pct = count/len(stats['confidence_scores'])*100
        print(f"   - Conf > {thresh}: {count} ({pct:.1f}%)")

# === VẼ BIỂU ĐỒ ===
print("\n5. VE BIEU DO PHAN BO...")

# Biểu đồ histogram confidence
plt.figure(figsize=(8, 6))
plt.hist(stats['confidence_scores'], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Confidence Score')
plt.ylabel('So luong')
plt.title('Phan bo Confidence Score cua Model')
plt.axvline(x=0.5, color='red', linestyle='--', label='Nguong 0.5')

if stats['confidence_scores']:
    mean_conf = np.mean(stats['confidence_scores'])
    plt.axvline(x=mean_conf, color='green', linestyle='-', label=f'TB: {mean_conf:.3f}')
plt.legend()

plt.tight_layout()
plt.savefig('ket_qua_thi_nghiem.png')
plt.show()

# === LƯU KẾT QUẢ ===
with open("ket_qua_thi_nghiem.txt", "w", encoding="utf-8") as f:
    f.write("KET QUA THI NGHIEM DANH GIA MODEL\n")
    f.write("="*50 + "\n")
    f.write(f"Thoi gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("1. THONG KE CO BAN:\n")
    f.write(f"   - Tong so frames: {stats['total_frames']}\n")
    f.write(f"   - Tong detections: {stats['detections_in_roi']}\n")
    f.write(f"   - TB detections/frame: {stats['detections_in_roi']/stats['total_frames']:.2f}\n\n")
    
    if stats['confidence_scores']:
        f.write("2. CONFIDENCE SCORE:\n")
        f.write(f"   - Trung binh: {np.mean(stats['confidence_scores']):.3f}\n")
        f.write(f"   - Trung vi: {np.median(stats['confidence_scores']):.3f}\n")
        f.write(f"   - Do lech chuan: {np.std(stats['confidence_scores']):.3f}\n")
        f.write(f"   - Min: {np.min(stats['confidence_scores']):.3f}\n")
        f.write(f"   - Max: {np.max(stats['confidence_scores']):.3f}\n\n")
        
        f.write("3. PHAN BO CONFIDENCE THEO NGUONG:\n")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            count = sum(1 for s in stats['confidence_scores'] if s > thresh)
            pct = count/len(stats['confidence_scores'])*100
            f.write(f"   - Conf > {thresh}: {count} ({pct:.1f}%)\n")

print("\nDa luu ket qua vao file:")
print("   - ket_qua_thi_nghiem.txt (so lieu)")
print("   - ket_qua_thi_nghiem.png (bieu do)")
print("\nDa hien thi video truc quan voi mau sac theo confidence")