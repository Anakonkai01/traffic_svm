import numpy as np 
import cv2
import os 
import glob 
import random 

# =============================================================================
# CẤU HÌNH HOG VÀ KÍCH THƯỚC
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
# HÀM TRÍCH XUẤT ĐẶC TRƯNG (Đã cải tiến)
# =============================================================================

def extract_hog_features(image_gray):
    """Trích xuất HOG từ một ảnh xám đã được tải."""
    try:
        image_resized = cv2.resize(image_gray, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        features = hog.compute(image_resized)
        return features.flatten()
    except Exception as e:
        # Bắt lỗi nếu resize hoặc compute thất bại
        print(f"  Lỗi khi trích xuất HOG: {e}")
        return None

def load_images_and_extract_features(path):
    """Tải ảnh và trích xuất HOG, xử lý lỗi nếu ảnh hỏng."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # *** CẢI TIẾN 1: Kiểm tra None ngay lập tức ***
    if img is None:
        print(f"Warning: Không thể đọc ảnh {path}. Bỏ qua.")
        return None
        
    return extract_hog_features(img)

# =============================================================================
# HÀM HUẤN LUYỆN SVM (Đã cải tiến)
# =============================================================================

def train_svm_detector(positive_image_dir: str, negative_image_dir: str, model_save_path: str):
    print("Training SVM detector (Sign or Not Sign))")
    
    features = []
    labels = []
    
    positive_image_paths = glob.glob(os.path.join(positive_image_dir, '*.*'))
    negative_image_paths = glob.glob(os.path.join(negative_image_dir, '*.*'))
    
    # *** CẢI TIẾN 2: Đếm chính xác ***
    pos_count = 0
    print("Loading positive samples...")
    for img_path in positive_image_paths:
        hog_features = load_images_and_extract_features(img_path)
        if hog_features is not None:
            features.append(hog_features)
            labels.append(1)  # Positive label
            pos_count += 1
    print(f"Number of positive samples: {pos_count} (từ {len(positive_image_paths)} files)")

    neg_count = 0
    print("Loading negative samples...")
    for img_path in negative_image_paths:
        hog_features = load_images_and_extract_features(img_path)
        if hog_features is not None:
            features.append(hog_features)
            labels.append(0)  # Negative label
            neg_count += 1
    print(f"Number of negative samples: {neg_count} (từ {len(negative_image_paths)} files)")
    
    # Kiểm tra nếu có đủ dữ liệu
    if pos_count == 0 or neg_count == 0:
        print("Lỗi: Cần có cả dữ liệu positive và negative để huấn luyện.")
        return

    # shuffle data 
    print("\nShuffling data...")
    combined = list(zip(features, labels))
    random.shuffle(combined)
    features, labels = zip(*combined)
    
    # convert to numpy arrays
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"Total samples for training: {len(labels)}")
    
    
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

   
    # Save the trained model 
    svm.save(model_save_path)
    print(f"✓ SVM model saved to {model_save_path}")
    
    
# run 
if __name__ == "__main__":
    positive_image_dir = "data_detector/positives"  # Thay bằng thư mục seed/final của bạn
    negative_image_dir = "data_detector/negatives"  # Thay bằng thư mục seed/final của bạn
    model_save_path = "svm_sign_detector_v4.xml"     # Path to save the trained model
    
    train_svm_detector(positive_image_dir, negative_image_dir, model_save_path)