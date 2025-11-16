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
# CẤU HÌNH HUẤN LUYỆN (QUAN TRỌNG)
# =============================================================================
# Giữ nguyên con số mà bạn đã chạy thành công (ví dụ: 300000)
# vì chúng ta vẫn cần dữ liệu vừa với RAM
MAX_TOTAL_SAMPLES = 10000000

# =============================================================================
# HÀM TRÍCH XUẤT HOG MÀU
# =============================================================================
def load_images_and_extract_features(path: str) -> Optional[np.ndarray]:
    """Tải ảnh MÀU, trích xuất HOG cho từng kênh B, G, R, và nối lại."""
    img_color = cv2.imread(path, cv2.IMREAD_COLOR) 
    if img_color is None:
        # print(f"Warning: Không thể đọc ảnh {path}. Bỏ qua.")
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
        # print(f"  Lỗi khi trích xuất HOG Màu từ {path}: {e}")
        return None

# =============================================================================
# HÀM HUẤN LUYỆN SVM (PHIÊN BẢN ÉP OVERFIT)
# =============================================================================
def train_svm_detector(positive_image_dir: str, negative_image_dir: str, model_save_path: str):
    print("="*70)
    print("  BẮT ĐẦU HUẤN LUYỆN MODEL (OpenCV - Color HOG + RBF)")
    print("  (PHIÊN BẢN ÉP OVERFIT VỚI GRID TÙY CHỈNH)")
    print("="*70)
    
    # --- Bước 1: Đọc tất cả đường dẫn (nhẹ) ---
    all_positive_paths = glob.glob(os.path.join(positive_image_dir, '*.*'))
    all_negative_paths = glob.glob(os.path.join(negative_image_dir, '*.*'))
    
    total_pos_available = len(all_positive_paths)
    total_neg_available = len(all_negative_paths)
    total_available = total_pos_available + total_neg_available
    
    if total_pos_available == 0 or total_neg_available == 0:
        print("Lỗi: Không tìm thấy ảnh positive hoặc negative.")
        return
        
    print(f"Tìm thấy tổng cộng {total_available} ảnh (Pos: {total_pos_available}, Neg: {total_neg_available})")

    # --- Bước 2: Tính toán số lượng mẫu để lấy (Sub-sampling) ---
    pos_ratio = total_pos_available / total_available
    neg_ratio = total_neg_available / total_available
    
    if total_available <= MAX_TOTAL_SAMPLES:
        num_pos_to_take = total_pos_available
        num_neg_to_take = total_neg_available
        print(f"Tổng data ({total_available}) nhỏ hơn MAX ({MAX_TOTAL_SAMPLES}). Huấn luyện trên tất cả.")
    else:
        num_pos_to_take = int(MAX_TOTAL_SAMPLES * pos_ratio)
        num_neg_to_take = int(MAX_TOTAL_SAMPLES * neg_ratio)
        print(f"Tổng data ({total_available}) lớn hơn MAX ({MAX_TOTAL_SAMPLES}).")
        print(f"Sẽ lấy ngẫu nhiên {MAX_TOTAL_SAMPLES} mẫu (giữ tỷ lệ).")
        
    print(f" -> Số mẫu Positive sẽ huấn luyện: {num_pos_to_take}")
    print(f" -> Số mẫu Negative sẽ huấn luyện: {num_neg_to_take}")
    
    # --- Bước 3: Lấy ngẫu nhiên các đường dẫn ---
    random.shuffle(all_positive_paths)
    random.shuffle(all_negative_paths)
    
    final_pos_paths = all_positive_paths[:num_pos_to_take]
    final_neg_paths = all_negative_paths[:num_neg_to_take]
    
    total_samples_to_train = len(final_pos_paths) + len(final_neg_paths)
    
    # --- Bước 4: Lấy kích thước vector ---
    print("\nĐang kiểm tra kích thước vector đặc trưng...")
    sample_features = load_images_and_extract_features(final_pos_paths[0])
    if sample_features is None:
        print(f"Lỗi: Không thể đọc ảnh mẫu {final_pos_paths[0]}")
        return
        
    feature_size = sample_features.shape[0]
    print(f"Kích thước vector HOG Màu: {feature_size}")

    # --- Bước 5: Cấp phát trước bộ nhớ (chỉ cho tập con) ---
    print(f"Cấp phát trước bộ nhớ cho {total_samples_to_train} mẫu...")
    features_np = np.zeros((total_samples_to_train, feature_size), dtype=np.float32)
    labels_np = np.zeros(total_samples_to_train, dtype=np.int32)
    
    current_index = 0
    
    # --- Bước 6: Load Positive ---
    print(f"Loading {len(final_pos_paths)} positive samples...")
    for i, img_path in enumerate(final_pos_paths):
        if (i + 1) % 100 == 0:
            print(f"    ... Đã xử lý {i + 1} / {len(final_pos_paths)} ảnh positive...")
        hog_features = load_images_and_extract_features(img_path)
        if hog_features is not None:
            features_np[current_index] = hog_features
            labels_np[current_index] = 1
            current_index += 1

    # --- Bước 7: Load Negative ---
    print(f"Loading {len(final_neg_paths)} negative samples...")
    for i, img_path in enumerate(final_neg_paths):
        if (i + 1) % 100 == 0:
            print(f"    ... Đã xử lý {i + 1} / {len(final_neg_paths)} ảnh negative...")
        hog_features = load_images_and_extract_features(img_path)
        if hog_features is not None:
            features_np[current_index] = hog_features
            labels_np[current_index] = 0
            current_index += 1
    
    total_valid_samples = current_index
    print(f"\nTổng số mẫu hợp lệ: {total_valid_samples}")

    # --- Bước 8: Cắt mảng về kích thước thật ---
    features = features_np[:total_valid_samples]
    labels = labels_np[:total_valid_samples]
    del features_np, labels_np # Giải phóng bộ nhớ mảng lớn ban đầu

    # --- Bước 9: Xáo trộn ---
    print("Shuffling data (sử dụng indices)...")
    indices = np.arange(total_valid_samples)
    random.shuffle(indices)
    
    features = features[indices]
    labels = labels[indices]
    
    print(f"Total samples for training: {len(labels)}")

    # ===============================================================
    # === PHẦN THAY ĐỔI ĐỂ ÉP OVERFIT ===
    # ===============================================================
    
    print("\nConfiguring SVM (OpenCV) for OVERFITTING (RBF Kernel)...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF) 
    
    print("Đang cấu hình Grid Search để ưu tiên Overfitting...")
    
    # Grid cho C: Ưu tiên các giá trị C RẤT LỚN
    c_grid_params = cv2.ml.ParamGrid_create()
    c_grid_params.minVal = 100  # Bắt đầu từ C=100 (rất cao)
    c_grid_params.maxVal = 10000 # Lên đến C=10,000
    c_grid_params.logStep = 10 # Bước nhảy log: 100, 1000, 10000
    
    # Grid cho Gamma: Ưu tiên các giá trị Gamma LỚN
    gamma_grid_params = cv2.ml.ParamGrid_create()
    gamma_grid_params.minVal = 1 # Bắt đầu từ Gamma=1 (rất cao)
    gamma_grid_params.maxVal = 100 # Lên đến Gamma=100
    gamma_grid_params.logStep = 10 # Bước nhảy log: 1, 10, 100
    
    print(f"  -> Sẽ tìm C trong khoảng [${c_grid_params.minVal}$, ${c_grid_params.maxVal}$]")
    print(f"  -> Sẽ tìm Gamma trong khoảng [${gamma_grid_params.minVal}$, ${gamma_grid_params.maxVal}$]")

    # Sử dụng kFold=2 de overfit =)) 
    kFold_val = 2 

    print(f"Starting SVM Auto-Training (kFold=${kFold_val}$) với Grid tùy chỉnh...")
    print("Bắt đầu huấn luyện... Việc này sẽ chậm, hãy kiên nhẫn.")

    svm.trainAuto(
        features, # Dữ liệu của bạn
        cv2.ml.ROW_SAMPLE, 
        labels, # Nhãn của bạn
        kFold=kFold_val,
        Cgrid=c_grid_params,      # <-- Ép C cao
        gammaGrid=gamma_grid_params # <-- Ép Gamma cao
    )
    
    print("\n✓ Auto-Training complete!")
    print(f"  -> Best Kernel: RBF")
    print(f"  -> Best C (Giá trị C tốt nhất): {svm.getC()}")
    print(f"  -> Best Gamma (Giá trị Gamma tốt nhất): {svm.getGamma()}")
   
    # --- Lưu model (XML) ---
    svm.save(model_save_path)
    print(f"\n✓ SVM model (OpenCV XML - Overfit) saved to {model_save_path}")
    print("="*70)
    
# =============================================================================
# CHẠY
# =============================================================================
if __name__ == "__main__":
    positive_image_dir = "../data_detector/positives"
    negative_image_dir = "../data_detector/negatives"
    
    # Đặt tên model mới để không ghi đè lên model cũ
    model_save_path = "models/svm_detector_model_color_rbf_OVERFIT_v1.xml" 
    
    train_svm_detector(positive_image_dir, negative_image_dir, model_save_path)