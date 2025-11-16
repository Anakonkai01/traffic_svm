import numpy as np 
import cv2
import os 
import glob 
import random 
from typing import Optional

# =============================================================================
# CẤU HÌNH HOG MÀU (PHẢI GIỐNG HỆT Ở MỌI NƠI)
# =============================================================================
RESIZE_DIM = (64, 64)
winSize = RESIZE_DIM
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# =============================================================================
# HÀM TRÍCH XUẤT HOG MÀU
# =============================================================================
def load_images_and_extract_features(path: str) -> Optional[np.ndarray]:
    """Tải ảnh MÀU, trích xuất HOG cho từng kênh B, G, R, và nối lại."""
    img_color = cv2.imread(path, cv2.IMREAD_COLOR) 
    if img_color is None:
        print(f"Warning: Không thể đọc ảnh {path}. Bỏ qua.")
        return None
    try:
        img_resized = cv2.resize(img_color, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        b_channel, g_channel, r_channel = cv2.split(img_resized)
        
        features_b = hog.compute(b_channel).flatten()
        features_g = hog.compute(g_channel).flatten()
        features_r = hog.compute(r_channel).flatten()
        
        final_features = np.hstack((features_b, features_g, features_r))
        return final_features
    except Exception as e:
        print(f"  Lỗi khi trích xuất HOG Màu từ {path}: {e}")
        return None

# =============================================================================
# HÀM HUẤN LUYỆN SVM (MODEL B - NÂNG CẤP)
# =============================================================================
def train_svm_detector(positive_image_dir: str, negative_image_dir: str, model_save_path: str):
    print("="*70)
    print("  BẮT ĐẦU HUẤN LUYỆN MODEL B (DETECTOR) - CAO CẤP (Color HOG + RBF)")
    print("="*70)
    
    features = []
    labels = []
    
    positive_image_paths = glob.glob(os.path.join(positive_image_dir, '*.*'))
    negative_image_paths = glob.glob(os.path.join(negative_image_dir, '*.*'))
    
    # --- Label 1: Positive ---
    pos_count = 0
    print(f"Loading positive samples (Color HOG) từ: {positive_image_dir}")
    total_pos = len(positive_image_paths)
    for i, img_path in enumerate(positive_image_paths):
        if (i + 1) % 100 == 0:
            print(f"    ... Đã xử lý {i + 1} / {total_pos} ảnh positive...")
        hog_features = load_images_and_extract_features(img_path)
        if hog_features is not None:
            features.append(hog_features)
            labels.append(1)
            pos_count += 1
    print(f"  -> Đã tải {pos_count} / {total_pos} mẫu positive.")

    # --- Label 0: Negative ---
    neg_count = 0
    print(f"Loading negative samples (Color HOG) từ: {negative_image_dir}")
    total_neg = len(negative_image_paths)
    for i, img_path in enumerate(negative_image_paths):
        if (i + 1) % 100 == 0:
            print(f"    ... Đã xử lý {i + 1} / {total_neg} ảnh negative...")
        hog_features = load_images_and_extract_features(img_path)
        if hog_features is not None:
            features.append(hog_features)
            labels.append(0)
            neg_count += 1
    print(f"  -> Đã tải {neg_count} / {total_neg} mẫu negative.")
    
    if pos_count == 0 or neg_count == 0:
        print("Lỗi: Cần có cả dữ liệu positive và negative để huấn luyện.")
        return

    # --- Xáo trộn ---
    print("\nShuffling data...")
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"Total samples for training: {len(labels)}")
    if len(features) > 0:
        print(f"Kích thước vector HOG Màu: {features.shape[1]}")

    # --- CẤU HÌNH VÀ HUẤN LUYỆN (ĐÃ THAY ĐỔI) ---
    print("\nConfiguring SVM for HIGH ACCURACY (RBF Kernel)...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    # 1. Sử dụng Kernel RBF (mạnh mẽ hơn LINEAR)
    svm.setKernel(cv2.ml.SVM_RBF) 
    
    print("Starting SVM Auto-Training (Grid Search)...")
    print("⚠️ CẢNH BÁO: Việc này sẽ CỰC KỲ CHẬM (có thể vài giờ),")
    print("   nhưng sẽ cho độ chính xác cao hơn.")
    print("   OpenCV không hiển thị tiến độ, hãy kiên nhẫn...")

    # 2. Sử dụng trainAuto để tự tìm C và Gamma tốt nhất
    # kFold=5 nghĩa là dùng Cross-Validation 5 lần.
    # Tăng kFold (ví dụ: 10) sẽ chính xác hơn nhưng chậm hơn nữa.
    svm.trainAuto(
        features, 
        cv2.ml.ROW_SAMPLE, 
        labels,
        kFold=10
    )
    
    print("\n✓ Auto-Training complete!")
    print(f"  -> Best Kernel: RBF")
    print(f"  -> Best C (Giá trị C tốt nhất): {svm.getC()}")
    print(f"  -> Best Gamma (Giá trị Gamma tốt nhất): {svm.getGamma()}")
   
    # --- Lưu model ---
    svm.save(model_save_path)
    print(f"\n✓ SVM model (Color HOG + RBF) saved to {model_save_path}")
    print("="*70)
    
# =============================================================================
# CHẠY
# =============================================================================
if __name__ == "__main__":
    positive_image_dir = "data_detector/positives"
    negative_image_dir = "data_detector/negatives"
    # Đặt tên mới để phân biệt với model cũ
    model_save_path = "models/svm_detector_model_color_rbf_v1.xml" 
    
    train_svm_detector(positive_image_dir, negative_image_dir, model_save_path)