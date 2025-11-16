import cv2
import numpy as np
import os
import glob
import random
from typing import Optional

# =============================================================================
# CẤU HÌNH HOG VÀ KÍCH THƯỚC (PHẢI GIỐNG HỆT Ở MỌI NƠI)
# =============================================================================
RESIZE_DIM = (64, 64) 
winSize = RESIZE_DIM
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

# Khởi tạo HOG
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# =============================================================================
# HÀM TRÍCH XUẤT ĐẶC TRƯNG
# =============================================================================
def get_hog_features(image_path: str) -> Optional[np.ndarray]:
    """Tải ảnh, chuyển xám, resize, trích xuất HOG, và xử lý lỗi."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Không thể đọc ảnh {image_path}. Bỏ qua.")
            return None
        
        img_resized = cv2.resize(img, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        features = hog.compute(img_resized)
        return features.flatten()
    except Exception as e:
        print(f"  Lỗi khi trích xuất HOG từ {image_path}: {e}")
        return None

# =============================================================================
# HÀM HUẤN LUYỆN SVM (MODEL A)
# =============================================================================
def train_recognizer_model(dataset_dir: str, model_save_path: str):
    print("=" * 70)
    print("  BẮT ĐẦU HUẤN LUYỆN MODEL A (RECOGNIZER: Multi-class)")
    print("=" * 70)
    
    features_list = []
    labels_list = []
    label_map = {} # Để lưu tên các lớp

    # Lấy danh sách các thư mục con (ví dụ: '0_stop', '1_turn_left', ...)
    class_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if not class_dirs:
        print(f"Lỗi: Không tìm thấy thư mục lớp nào trong {dataset_dir}")
        return

    print("Đang tìm các lớp...")
    
    # Duyệt qua từng thư mục lớp
    for class_name in class_dirs:
        try:
            # Trích xuất ID lớp từ tên thư mục (ví dụ: "0_stop" -> 0)
            label_id = int(class_name.split('_')[0])
            label_map[label_id] = class_name # Lưu lại (0: "0_stop")
        except Exception:
            print(f"Warning: Bỏ qua thư mục có tên không hợp lệ: {class_name}")
            continue

        class_path = os.path.join(dataset_dir, class_name)
        print(f"\nĐang xử lý Lớp: {class_name} (ID: {label_id})")
        
        image_paths = glob.glob(os.path.join(class_path, "*.*"))
        if not image_paths:
            print("  Warning: Không có ảnh trong thư mục này.")
            continue
            
        class_features_count = 0
        for img_path in image_paths:
            hog_features = get_hog_features(img_path)
            
            if hog_features is not None:
                features_list.append(hog_features)
                labels_list.append(label_id) # Gán nhãn là ID (0, 1, 2,...)
                class_features_count += 1
                
        print(f"  Đã trích xuất đặc trưng từ {class_features_count} ảnh.")

    if not features_list:
        print("Lỗi nghiêm trọng: Không có dữ liệu nào được tải để huấn luyện.")
        return

    # Xáo trộn dữ liệu
    print("\nĐang kết hợp và xáo trộn dữ liệu...")
    combined = list(zip(features_list, labels_list))
    random.shuffle(combined)
    features_list, labels_list = zip(*combined)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    
    print(f"\nTổng cộng {len(features)} mẫu từ {len(class_dirs)} lớp.")

    # Cấu hình và Huấn luyện SVM
    print("\nĐang cấu hình SVM...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF) # using RBF kernel
    
    # Create and train SVM
    print("\nConfiguring SVM...")
    print("Start Auto-Train SVM (Grid Search) ..")
    
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF) # SVM_RBR 
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.trainAuto(
        features,
        cv2.ml.ROW_SAMPLE,
        labels,
        kFold=5,
    )
    
    print("SVM training completed.")
    print(f"Best C: {svm.getC()}")
    print(f"Best Gamma: {svm.getGamma()}")
    
    # Lưu mô hình
    svm.save(model_save_path)
    print(f"✓ Mô hình Recognizer đã được lưu tại: {model_save_path}")

    print("\n" + "=" * 70)
    print("  BẢN ĐỒ NHÃN (LABEL MAP) - HÃY LƯU LẠI CÁI NÀY!")
    print("  Dùng nó để tạo Dictionary trong file Colab của bạn.")
    print("=" * 70)
    for label_id, class_name in sorted(label_map.items()):
        print(f"  ID: {label_id}  => Tên thư mục: {class_name}")
    print("=" * 70)

# =============================================================================
# CHẠY
# =============================================================================
if __name__ == "__main__":
    # ĐƯỜNG DẪN ĐẦU VÀO
    # Thư mục chứa các thư mục con đã phân loại (ví dụ: '0_stop', '1_turn_left')
    DATASET_DIR = "data_recognizer" 
    
    # ĐƯỜNG DẪN ĐẦU RA
    MODEL_PATH = "svm_sign_recognizer_v3.xml" # Tên model A
    
    if not os.path.isdir(DATASET_DIR):
        print(f"Lỗi: Không tìm thấy thư mục '{DATASET_DIR}'")
    else:
        train_recognizer_model(DATASET_DIR, MODEL_PATH)