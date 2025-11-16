import cv2
import numpy as np
import os
import glob
import shutil
from sklearn.cluster import KMeans
from typing import Optional, List, Tuple

# =============================================================================
# CÀI ĐẶT
# =============================================================================
# Thư mục nguồn (chứa TẤT CẢ các biển báo đã được lọc)
SOURCE_DIR = "data_detector/positives" 

# Thư mục đầu ra (nơi chứa các cụm)
OUTPUT_DIR = "dataset_recognizer_clustered"

# !!! THAM SỐ QUAN TRỌNG NHẤT !!!
# Bạn nghĩ có bao nhiêu LOẠI biển báo trong thư mục SOURCE_DIR?
# Hãy đếm sơ qua và đặt con số đó ở đây.
NUM_CLUSTERS = 20 # <-- HÃY THAY ĐỔI SỐ NÀY (ví dụ: 5, 8, 15)

# =============================================================================
# CẤU HÌNH HOG (PHẢI GIỐNG HỆT CÁC FILE KHÁC)
# =============================================================================
RESIZE_DIM = (64, 64)
winSize = RESIZE_DIM
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# =============================================================================
# HÀM TRÍCH XUẤT HOG
# =============================================================================
def get_hog_features(image_path: str) -> Optional[np.ndarray]:
    """Đọc ảnh, trích xuất HOG."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        img_resized = cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        features = hog.compute(img_resized)
        return features.flatten()
    except Exception:
        return None

# =============================================================================
# HÀM CHÍNH: PHÂN CỤM
# =============================================================================
def cluster_images():
    print("="*70)
    print("  BẮT ĐẦU PHÂN CỤM ẢNH (K-MEANS + HOG)")
    print("="*70)
    
    # 1. Trích xuất đặc trưng HOG cho tất cả ảnh
    print(f"Đang quét {SOURCE_DIR} và trích xuất đặc trưng HOG...")
    
    features_list = []
    image_path_list = [] # Lưu lại đường dẫn để copy file sau
    
    all_image_paths = glob.glob(os.path.join(SOURCE_DIR, "*.*"))
    if not all_image_paths:
        print(f"Lỗi: Không tìm thấy ảnh nào trong {SOURCE_DIR}")
        return

    for img_path in all_image_paths:
        features = get_hog_features(img_path)
        if features is not None:
            features_list.append(features)
            image_path_list.append(img_path)
            
    if not features_list:
        print("Lỗi: Không trích xuất được đặc trưng HOG nào.")
        return
        
    features_matrix = np.array(features_list, dtype=np.float32)
    print(f"  Đã trích xuất đặc trưng cho {len(image_path_list)} ảnh.")
    
    # 2. Chạy thuật toán K-Means
    print(f"\nĐang chạy K-Means để chia thành {NUM_CLUSTERS} cụm...")
    print("(Việc này có thể mất vài phút nếu dữ liệu lớn)")
    
    # n_init=10 để chạy 10 lần và chọn kết quả tốt nhất
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=20, max_iter=1000)
    kmeans.fit(features_matrix)
    
    # labels_ là một mảng, mỗi phần tử là ID cụm (từ 0 đến K-1)
    labels = kmeans.labels_
    print("✓ Phân cụm hoàn tất.")
    
    # 3. Tạo thư mục và copy ảnh
    print(f"\nĐang tạo thư mục đầu ra tại: {OUTPUT_DIR}")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # Xóa thư mục cũ nếu có
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Tạo các thư mục con cho từng cụm
    for i in range(NUM_CLUSTERS):
        os.makedirs(os.path.join(OUTPUT_DIR, f"cluster_{i}"), exist_ok=True)
        
    print("Đang sao chép ảnh vào các thư mục cụm...")
    for img_path, label in zip(image_path_list, labels):
        # Lấy tên file
        filename = os.path.basename(img_path)
        # Tạo đường dẫn đích
        dest_dir = os.path.join(OUTPUT_DIR, f"cluster_{label}")
        # Copy file
        shutil.copy(img_path, os.path.join(dest_dir, filename))
        
    print("\n" + "="*70)
    print("✓ HOÀN TẤT!")
    print(f"Đã tạo {NUM_CLUSTERS} thư mục con trong '{OUTPUT_DIR}'")
    print("="*70)
    print("\nVIỆC CỦA BẠN BÂY GIỜ:")
    print("1. Mở thư mục 'dataset_recognizer_clustered'.")
    print("2. Xem từng thư mục 'cluster_0', 'cluster_1', ...")
    print("3. Đổi tên chúng (ví dụ: 'cluster_0' chứa toàn biển Stop -> đổi tên thành '0_stop').")
    print("4. Dọn dẹp các ảnh bị lạc (nếu có).")
    print("5. Thư mục này chính là 'dataset_recognizer_custom' của bạn, sẵn sàng để huấn luyện!")
    print("="*70)
    
# =============================================================================
# CHẠY CHƯƠNG TRÌNH
# =============================================================================
if __name__ == "__main__":
    cluster_images()