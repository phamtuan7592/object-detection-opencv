# Hệ thống phát hiện đối tượng với ROI

Dự án phát hiện đối tượng sử dụng mô hình YOLO (hoặc tương tự) và hỗ trợ vùng quan tâm (ROI) trên webcam hoặc file video.

## Yêu cầu hệ thống

- Python 3.10 hoặc mới hơn
- Webcam (nếu chạy với nguồn webcam)

## Cài đặt môi trường

### B.1 Cài đặt môi trường

1. **Tải và cài đặt Python 3.10+**  
   Truy cập [python.org](https://python.org) và tải Python phù hợp với hệ điều hành của bạn.

2. **Mở Terminal tại thư mục dự án**  
   (Trên Windows có thể mở Command Prompt hoặc PowerShell tại đây bằng cách gõ `cmd` vào thanh địa chỉ file explorer)

3. **Cài đặt các thư viện cần thiết**  
   Chạy lệnh sau để cài đặt tất cả thư viện được liệt kê trong `requirements.txt`:
   ```bash
   pip install -r requirements.txt
