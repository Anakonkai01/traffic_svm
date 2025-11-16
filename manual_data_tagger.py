import cv2
import os
import uuid # Để tạo tên file duy nhất

# =============================================================================
# CÀI ĐẶT
# =============================================================================
VIDEO_SOURCE = 'video/video2.mp4' # <-- THAY ĐỔI VIDEO Ở ĐÂY (video1 hoặc video2)
RESIZE_DIM = (64, 64) # Kích thước chuẩn để lưu (phải giống HOG)

OUTPUT_POS_DIR = "dataset_detector/positive"
OUTPUT_NEG_DIR = "dataset_detector/negative"

# Đảm bảo thư mục tồn tại
os.makedirs(OUTPUT_POS_DIR, exist_ok=True)
os.makedirs(OUTPUT_NEG_DIR, exist_ok=True)

# Biến toàn cục để lưu tọa độ chuột
drawing = False
ix, iy = -1, -1
rect = (0, 0, 0, 0) # (x, y, w, h)
frame_copy = None # Để vẽ lên

# =============================================================================
# HÀM XỬ LÝ CHUỘT
# =============================================================================
def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, rect, frame_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy() # Reset lại frame
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Lưu hình chữ nhật cuối cùng
        x_start, y_start = min(ix, x), min(iy, y)
        x_end, y_end = max(ix, x), max(iy, y)
        rect = (x_start, y_start, x_end - x_start, y_end - y_start)
        cv2.rectangle(frame_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

# =============================================================================
# HÀM CHÍNH
# =============================================================================
print("Mở video...")
cap = cv2.VideoCapture(VIDEO_SOURCE)
cv2.namedWindow('Tagger Tool')
cv2.setMouseCallback('Tagger Tool', draw_rect)

print("="*50)
print("  HƯỚNG DẪN SỬ DỤNG TOOL")
print("="*50)
print(" - Kéo chuột để vẽ một hộp.")
print(" - Nhấn 's' = Lưu là Positive (Sign)")
print(" - Nhấn 'n' = Lưu là Negative (Not-Sign)")
print(" - Nhấn [Spacebar] = Frame tiếp theo")
print(" - Nhấn 'q' = Thoát")
print("="*50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Hết video.")
        break
    
    frame_copy = frame.copy() # Bản sao để vẽ
    
    while True:
        cv2.imshow('Tagger Tool', frame_copy)
        key = cv2.waitKey(1) & 0xFF
        
        # Thoát
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
        # Frame tiếp theo
        elif key == ord(' '):
            rect = (0, 0, 0, 0) # Reset hộp
            break # Thoát vòng lặp trong để đọc frame mới
            
        # Lưu Positive
        elif key == ord('s'):
            x, y, w, h = rect
            if w > 0 and h > 0:
                cropped = frame[y:y+h, x:x+w]
                resized = cv2.resize(cropped, RESIZE_DIM, interpolation=cv2.INTER_AREA)
                filename = f"pos_{uuid.uuid4().hex[:6]}.png"
                cv2.imwrite(os.path.join(OUTPUT_POS_DIR, filename), resized)
                print(f"[ĐÃ LƯU POSITIVE]: {filename}")
                
        # Lưu Negative
        elif key == ord('n'):
            x, y, w, h = rect
            if w > 0 and h > 0:
                cropped = frame[y:y+h, x:x+w]
                resized = cv2.resize(cropped, RESIZE_DIM, interpolation=cv2.INTER_AREA)
                filename = f"neg_{uuid.uuid4().hex[:6]}.png"
                cv2.imwrite(os.path.join(OUTPUT_NEG_DIR, filename), resized)
                print(f"[ĐÃ LƯU NEGATIVE]: {filename}")

cap.release()
cv2.destroyAllWindows()