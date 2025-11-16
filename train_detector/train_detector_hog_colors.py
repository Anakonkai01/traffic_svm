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
# HÀM HUẤN LUYỆN SVM (MODEL B - NÂNG CẤP VÀ TỐI ƯU RAM)
# =============================================================================
def train_svm_detector(positive_image_dir: str, negative_image_dir: str, model_save_path: str):
    print("="*70)
    print("  BẮT ĐẦU HUẤN LUYỆN MODEL B (DETECTOR) - CAO CẤP (Color HOG + RBF)")
    print("  (PHIÊN BẢN TỐI ƯU RAM)")
    print("="*70)
    
    positive_image_paths = glob.glob(os.path.join(positive_image_dir, '*.*'))
    negative_image_paths = glob.glob(os.path.join(negative_image_dir, '*.*'))
    
    total_pos = len(positive_image_paths)
    total_neg = len(negative_image_paths)
    total_samples_max = total_pos + total_neg

    if total_pos == 0 or total_neg == 0:
        print("Lỗi: Không tìm thấy ảnh positive hoặc negative.")
        return

    # --- Lấy kích thước vector đặc trưng ---
    print("Đang kiểm tra kích thước vector đặc trưng...")
    sample_features = load_images_and_extract_features(positive_image_paths[0])
    if sample_features is None:
        print(f"Lỗi: Không thể đọc ảnh mẫu: {positive_image_paths[0]}")
        return
    
    feature_size = sample_features.shape[0]
    print(f"Kích thước vector HOG Màu: {feature_size} (1764 * 3 = 5292)")

    # --- TỐI ƯU RAM: CẤP PHÁT TRƯỚC (PRE-ALLOCATION) ---
    # Thay vì dùng list.append, chúng ta tạo 2 mảng NumPy lớn ngay từ đầu
    print(f"Cấp phát trước bộ nhớ cho {total_samples_max} mẫu...")
    features_np = np.zeros((total_samples_max, feature_size), dtype=np.float32)
    labels_np = np.zeros(total_samples_max, dtype=np.int32)
    
    current_index = 0
    
    # --- Label 1: Positive ---
    print(f"Loading positive samples (Color HOG) từ: {positive_image_dir}")
    pos_count = 0
    for i, img_path in enumerate(positive_image_paths):
        if (i + 1) % 100 == 0:
            print(f"    ... Đã xử lý {i + 1} / {total_pos} ảnh positive...")
        hog_features = load_images_and_extract_features(img_path)
        if hog_features is not None:
            features_np[current_index] = hog_features
            labels_np[current_index] = 1
            current_index += 1
            pos_count += 1
    print(f"  -> Đã tải {pos_count} / {total_pos} mẫu positive.")

    # --- Label 0: Negative ---
    print(f"Loading negative samples (Color HOG) từ: {negative_image_dir}")
    neg_count = 0
    for i, img_path in enumerate(negative_image_paths):
        if (i + 1) % 100 == 0:
            print(f"    ... Đã xử lý {i + 1} / {total_neg} ảnh negative...")
        hog_features = load_images_and_extract_features(img_path)
        if hog_features is not None:
            features_np[current_index] = hog_features
            labels_np[current_index] = 0
            current_index += 1
            neg_count += 1
    print(f"  -> Đã tải {neg_count} / {total_neg} mẫu negative.")
    
    total_valid_samples = current_index
    print(f"\nTổng số mẫu hợp lệ: {total_valid_samples} (từ {total_samples_max} tệp)")

    if pos_count == 0 or neg_count == 0:
        print("Lỗi: Cần có cả dữ liệu positive và negative để huấn luyện.")
        return

    # --- TỐI ƯU RAM: Cắt mảng về kích thước thật ---
    print("Cắt mảng về kích thước thực...")
    features = features_np[:total_valid_samples]
    labels = labels_np[:total_valid_samples]
    
    # Giải phóng bộ nhớ mảng lớn ban đầu
    del features_np, labels_np 

    # --- TỐI ƯU RAM: Xáo trộn bằng chỉ số (index) ---
    print("Shuffling data (sử dụng indices)...")
    indices = np.arange(total_valid_samples)
    random.shuffle(indices)
    
    features = features[indices]
    labels = labels[indices]
    
    print(f"Total samples for training: {len(labels)}")

    # --- CẤU HÌNH VÀ HUẤN LUYỆN ---
    print("\nConfiguring SVM for HIGH ACCURACY (RBF Kernel)...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF) 
    
    print("Starting SVM Auto-Training (Grid Search)...")
    print("⚠️ CẢNH BÁO: Việc này sẽ CỰC KỲ CHẬM (có thể vài giờ),")
    print("   nhưng sẽ cho độ chính xác cao hơn.")
    print("   OpenCV không hiển thị tiến độ, hãy kiên nhẫn...")

    svm.trainAuto(
        features, 
        cv2.ml.ROW_SAMPLE, 
        labels,
        kFold=3
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
    positive_image_dir = "../data_detector/positives"
    negative_image_dir = "../data_detector/negatives"
    # Đặt tên mới để phân biệt với model cũ
    model_save_path = "models/svm_detector_model_color_rbf_v1.xml" 
    
    train_svm_detector(positive_image_dir, negative_image_dir, model_save_path)