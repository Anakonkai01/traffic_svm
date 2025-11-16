import cv2
import numpy as np
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
# HÀM TRÍCH XUẤT HOG MÀU (Giống hệt file train_detector.py)
# =============================================================================
def get_color_hog_features(path: str) -> Optional[np.ndarray]:
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
# HÀM HUẤN LUYỆN SVM (MODEL A - NÂNG CẤP)
# =============================================================================
def train_recognizer_model(dataset_dir: str, model_save_path: str):
    print("=" * 70)
    print("  BẮT ĐẦU HUẤN LUYỆN MODEL A (RECOGNIZER) - CAO CẤP (Color HOG + RBF)")
    print("=" * 70)
    
    features_list = []
    labels_list = []
    label_map = {}

    class_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if not class_dirs:
        print(f"Lỗi: Không tìm thấy thư mục lớp nào trong {dataset_dir}")
        return

    print("Đang tìm các lớp...")
    
    for class_name in class_dirs:
        try:
            label_id = int(class_name.split('_')[0])
            label_map[label_id] = class_name
        except Exception:
            print(f"Warning: Bỏ qua thư mục có tên không hợp lệ: {class_name}")
            continue

        class_path = os.path.join(dataset_dir, class_name)
        print(f"\nĐang xử lý Lớp: {class_name} (ID: {label_id})")
        
        image_paths = glob.glob(os.path.join(class_path, "*.*"))
        total_imgs = len(image_paths)
        if total_imgs == 0:
            print("  Warning: Không có ảnh trong thư mục này.")
            continue
            
        class_features_count = 0
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 50 == 0:
                print(f"    ... Đã xử lý {i + 1} / {total_imgs} ảnh...")
            
            hog_features = get_color_hog_features(img_path) # <-- Gọi hàm mới
            
            if hog_features is not None:
                features_list.append(hog_features)
                labels_list.append(label_id)
                class_features_count += 1
                
        print(f"  Đã trích xuất đặc trưng từ {class_features_count} / {total_imgs} ảnh.")

    if not features_list:
        print("Lỗi nghiêm trọng: Không có dữ liệu nào được tải để huấn luyện.")
        return

    # --- Xáo trộn ---
    print("\nĐang kết hợp và xáo trộn dữ liệu...")
    combined = list(zip(features_list, labels_list))
    random.shuffle(combined)
    features_list, labels_list = zip(*combined)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    print(f"\nTổng cộng {len(features)} mẫu từ {len(class_dirs)} lớp.")
    if len(features) > 0:
        print(f"Kích thước vector HOG Màu: {features.shape[1]}")

    # --- CẤU HÌNH VÀ HUẤN LUYỆN (ĐÃ THAY ĐỔI) ---
    print("\nConfiguring SVM for HIGH ACCURACY (RBF Kernel)...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    # 1. Sử dụng Kernel RBF
    svm.setKernel(cv2.ml.SVM_RBF) 
    
    print("Starting SVM Auto-Training (Grid Search)...")
    print("⚠️ CẢNH BÁO: Việc này sẽ CỰC KỲ CHẬM.")
    print("   Hãy kiên nhẫn...")

    # 2. Sử dụng trainAuto
    svm.trainAuto(
        features, 
        cv2.ml.ROW_SAMPLE, 
        labels,
        kFold=10
    )
    
    print("\n✓ Auto-Training complete!")
    print(f"  -> Best Kernel: RBF")
    print(f"  -> Best C: {svm.getC()}")
    print(f"  -> Best Gamma: {svm.getGamma()}")
    
    # --- Lưu model ---
    svm.save(model_save_path)
    print(f"\n✓ Mô hình Recognizer (Color HOG + RBF) đã được lưu tại: {model_save_path}")

    # --- In Label Map ---
    print("\n" + "=" * 70)
    print("  BẢN ĐỒ NHÃN (LABEL MAP) - HÃY LƯU LẠI CÁI NÀY!")
    print("=" * 70)
    for label_id, class_name in sorted(label_map.items()):
        print(f"  ID: {label_id}  => Tên thư mục: {class_name}")
    print("=" * 70)

# =============================================================================
# CHẠY
# =============================================================================
if __name__ == "__main__":
    # Thư mục chứa các thư mục con đã phân loại (0_stop, 1_turn_left...)
    DATASET_DIR = "data_recognizer" 
    # Đặt tên mới
    MODEL_PATH = "models/svm_recognizer_model_color_rbf_v1.xml" 
    
    if not os.path.isdir(DATASET_DIR):
        print(f"Lỗi: Không tìm thấy thư mục '{DATASET_DIR}'")
    else:
        train_recognizer_model(DATASET_DIR, MODEL_PATH)