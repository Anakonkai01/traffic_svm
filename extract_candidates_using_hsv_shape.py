import cv2
import os
import uuid
# Chúng ta sẽ import các class đã viết từ file chính
from refactor_claude import TrafficSignConfig, TrafficSignDetector 

# =============================================================================
# CÀI ĐẶT
# =============================================================================
VIDEO_SOURCE = 'video/video2.mp4' # <-- THAY ĐỔI VIDEO Ở ĐÂY
OUTPUT_TEMP_DIR = "temp_candidates" # Thư mục tạm chứa tất cả ứng viên
RESIZE_DIM = (64, 64) # Kích thước chuẩn (phải giống HOG)

os.makedirs(OUTPUT_TEMP_DIR, exist_ok=True)

# =============================================================================
# HÀM CHÍNH
# =============================================================================
def generate_candidates():
    print("Bắt đầu trích xuất ứng viên (Bootstrapping)...")
    
    # 1. Khởi tạo Config và Detector (từ file refactor_claude.py)
    # RẤT QUAN TRỌNG: Đảm bảo bạn đã nới lỏng các tham số
    # (min_area, circularity, solidity) trong TrafficSignConfig
    config = TrafficSignConfig()
    detector = TrafficSignDetector(config)
    
    print(f"Đang đọc video: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    full_frame_dims = (w_orig, h_orig)
    
    frame_num = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 2. Chạy Giai đoạn 1 (DIP) để tìm ứng viên
        # (detections = danh sách các bbox tìm được bằng màu sắc + hình dạng)
        detections, _ = detector.process_frame(frame, full_frame_dims)
        
        if detections:
            for (bbox, color, metrics) in detections:
                x, y, w, h = bbox
                
                # Cắt (crop) và lưu ứng viên
                try:
                    cropped = frame[y:y+h, x:x+w]
                    resized = cv2.resize(cropped, RESIZE_DIM, interpolation=cv2.INTER_AREA)
                    
                    filename = f"cand_frame{frame_num}_{color}_{uuid.uuid4().hex[:4]}.png"
                    cv2.imwrite(os.path.join(OUTPUT_TEMP_DIR, filename), resized)
                    saved_count += 1
                except Exception as e:
                    pass # Bỏ qua nếu crop bị lỗi (ví dụ: bbox ở biên)
                    
        frame_num += 1
        if frame_num % 100 == 0:
            print(f"Đã xử lý {frame_num} frames... tìm thấy {saved_count} ứng viên.")
            
    cap.release()
    print("\n" + "="*50)
    print("✓ Trích xuất hoàn tất!")
    print(f"Tổng cộng {saved_count} ứng viên đã được lưu vào:")
    print(f"  -> {OUTPUT_TEMP_DIR}")
    print("\nVIỆC CỦA BẠN BÂY GIỜ:")
    print("1. Mở thư mục 'temp_candidates'.")
    print("2. Kéo các ảnh BIỂN BÁO THẬT vào 'dataset_detector/positive/'.")
    print("3. Kéo các ảnh LỖI (đèn, cây, đường) vào 'dataset_detector/negative/'.")
    print("="*50)

# =============================================================================
# CHẠY CHƯƠNG TRÌNH
# =============================================================================
if __name__ == "__main__":
    # Lưu ý: Cần 'nới lỏng' tham số trong file 'refactor_claude.py' trước khi chạy
    try:
        generate_candidates()
    except ImportError:
        print("\nLỖI: Không thể import 'TrafficSignConfig' từ 'refactor_claude.py'.")
        print("Hãy đảm bảo 2 file (.py) ở cùng một thư mục.")
    except Exception as e:
        print(f"Một lỗi đã xảy ra: {e}")