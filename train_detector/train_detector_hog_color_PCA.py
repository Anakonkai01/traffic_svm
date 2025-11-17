import numpy as np 
import cv2
import os 
import glob 
import random 
import gc

# =============================================================================
# CẤU HÌNH
# =============================================================================
RESIZE_DIM = (64, 64)
winSize = RESIZE_DIM
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# Số chiều muốn giữ lại (Nén từ 5292 -> 300)
PCA_COMPONENTS = 300 

# CỜ CẤU HÌNH QUAN TRỌNG
# Set = True: Tải file PCA đã có, bỏ qua bước tính toán.
# Set = False: Tính toán lại PCA từ đầu và lưu đè file cũ.
USE_PRECALCULATED_PCA = True 

# =============================================================================
# HÀM TRÍCH XUẤT HOG (Thô)
# =============================================================================
def extract_hog_raw(img_path):
    """Trích xuất HOG màu (5292 chiều)"""
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    if img_color is None: return None
    try:
        img_resized = cv2.resize(img_color, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        b, g, r = cv2.split(img_resized)
        f_b = hog.compute(b).flatten()
        f_g = hog.compute(g).flatten()
        f_r = hog.compute(r).flatten()
        return np.hstack((f_b, f_g, f_r))
    except: return None

# =============================================================================
# MAIN ROUTINE
# =============================================================================
def train_svm_with_pca(positive_dir, negative_dir, svm_model_path, pca_model_path):
    print("="*70)
    print("  TRAINING SVM VỚI COLOR HOG + PCA")
    print(f"  Chế độ PCA có sẵn: {'BẬT' if USE_PRECALCULATED_PCA else 'TẮT'}")
    print("="*70)

    all_pos = glob.glob(os.path.join(positive_dir, '*.*'))
    all_neg = glob.glob(os.path.join(negative_dir, '*.*'))
    all_paths = all_pos + all_neg
    total_files = len(all_paths)
    
    if total_files == 0:
        print("Lỗi: Không tìm thấy dữ liệu ảnh.")
        return

    # --- GIAI ĐOẠN 1: CHUẨN BỊ PCA ---
    mean = None
    eigenvectors = None

    # Kiểm tra xem file PCA có tồn tại không nếu đang bật chế độ dùng lại
    if USE_PRECALCULATED_PCA and not os.path.exists(pca_model_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file {pca_model_path} dù đã bật chế độ dùng lại.")
        print("   -> Chuyển sang tính toán lại PCA.")
        need_compute_pca = True
    else:
        need_compute_pca = not USE_PRECALCULATED_PCA

    if need_compute_pca:
        print("\n--- GIAI ĐOẠN 1: Tính toán ma trận PCA (Học cách nén) ---")
        # Lấy mẫu đại diện (khoảng 20.000 ảnh) để tính PCA
        pca_sample_size = min(20000, len(all_paths))
        print(f"Sử dụng {pca_sample_size} mẫu ngẫu nhiên để tính PCA...")
        
        pca_samples = random.sample(all_paths, pca_sample_size)
        pca_data = []
        
        for i, path in enumerate(pca_samples):
            feat = extract_hog_raw(path)
            if feat is not None:
                pca_data.append(feat)
            if i % 1000 == 0: print(f"  - Loading PCA samples: {i}/{pca_sample_size}", end='\r')
                
        pca_data = np.array(pca_data, dtype=np.float32)
        
        print(f"\nĐang tính toán PCA (Compute)... Data shape: {pca_data.shape}")
        mean, eigenvectors = cv2.PCACompute(pca_data, mean=None, maxComponents=PCA_COMPONENTS)
        print(f"  -> PCA hoàn tất. Đã nén xuống {eigenvectors.shape[0]} chiều.")
        
        # Lưu PCA
        print(f"Lưu PCA model vào {pca_model_path}...")
        fs = cv2.FileStorage(pca_model_path, cv2.FILE_STORAGE_WRITE)
        fs.write("mean", mean)
        fs.write("eigenvectors", eigenvectors)
        fs.release()
        
        del pca_data
        gc.collect()
    else:
        print(f"\n--- GIAI ĐOẠN 1: Tải PCA có sẵn từ {pca_model_path} ---")
        fs = cv2.FileStorage(pca_model_path, cv2.FILE_STORAGE_READ)
        mean = fs.getNode("mean").mat()
        eigenvectors = fs.getNode("eigenvectors").mat()
        fs.release()
        
        if mean is None or eigenvectors is None:
            print("❌ Lỗi: File PCA bị hỏng hoặc rỗng.")
            return
        print(f"  -> Đã tải xong. Số chiều nén: {eigenvectors.shape[0]}")

    # --- GIAI ĐOẠN 2: LOAD TOÀN BỘ DATA & NÉN NGAY LẬP TỨC ---
    print("\n--- GIAI ĐOẠN 2: Load và Nén TOÀN BỘ dữ liệu ---")
    print(f"Đang xử lý {total_files} ảnh...")
    
    # Cấp phát trước bộ nhớ (Rất nhẹ: N x 300)
    actual_components = eigenvectors.shape[0]
    X_train = np.zeros((total_files, actual_components), dtype=np.float32)
    y_train = np.zeros(total_files, dtype=np.int32)
    
    idx = 0
    
    # Load Positive
    for i, path in enumerate(all_pos):
        feat_raw = extract_hog_raw(path)
        if feat_raw is not None:
            # NÉN: (vector - mean) dot (eigenvectors.T)
            feat_pca = cv2.PCAProject(feat_raw.reshape(1, -1), mean, eigenvectors)
            X_train[idx] = feat_pca.flatten()
            y_train[idx] = 1
            idx += 1
        if i % 1000 == 0: print(f"  Pos: {i}/{len(all_pos)}", end='\r')

    # Load Negative
    for i, path in enumerate(all_neg):
        feat_raw = extract_hog_raw(path)
        if feat_raw is not None:
            feat_pca = cv2.PCAProject(feat_raw.reshape(1, -1), mean, eigenvectors)
            X_train[idx] = feat_pca.flatten()
            y_train[idx] = 0
            idx += 1
        if i % 1000 == 0: print(f"  Neg: {i}/{len(all_neg)}", end='\r')
        
    # Cắt phần thừa
    X_train = X_train[:idx]
    y_train = y_train[:idx]
    
    print(f"\n\nTổng số mẫu đã nén thành công: {idx}")
    print(f"Kích thước ma trận Training: {X_train.shape}")

    # --- GIAI ĐOẠN 3: TRAIN SVM ---
    print("\n--- GIAI ĐOẠN 3: Train SVM ---")
    print("Shuffling data...")
    indices = np.arange(idx)
    random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print("Configuring SVM (RBF Kernel)...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    
    print("Starting TrainAuto (Grid Search)...")
    # SỬA LỖI TẠI ĐÂY: Truyền trực tiếp mảng thay vì TrainData_create
    svm.trainAuto(
        X_train,                # Samples
        cv2.ml.ROW_SAMPLE,      # Layout
        y_train,                # Responses (Labels)
        kFold=5 
    )
    
    print("\n✓ Huấn luyện hoàn tất!")
    print(f"  -> Best C: {svm.getC()}")
    print(f"  -> Best Gamma: {svm.getGamma()}")
    
    svm.save(svm_model_path)
    print(f"✓ SVM Model saved to {svm_model_path}")
    print("="*70)

if __name__ == "__main__":
    # Đường dẫn thư mục dữ liệu
    pos_dir = "../data_detector/positives"
    neg_dir = "../data_detector/negatives"
    
    # Tạo thư mục models nếu chưa có
    if not os.path.exists("models"): os.makedirs("models")
    
    # Đường dẫn file output
    svm_path = "models/svm_detector_color_pca_v1.xml"
    pca_path = "models/pca_transform_v1.xml" 
    
    train_svm_with_pca(pos_dir, neg_dir, svm_path, pca_path)