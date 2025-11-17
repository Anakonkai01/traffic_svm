import numpy as np 
import cv2
import os 
import glob 
import random 
import gc

# =============================================================================
# CẤU HÌNH (PHẢI KHỚP VỚI FILE MAIN PIPELINE)
# =============================================================================
RESIZE_DIM = (64, 64)
winSize = RESIZE_DIM
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

# Số chiều nén (Giữ lại 300 đặc trưng quan trọng nhất)
PCA_COMPONENTS = 300 

# Cờ cấu hình:
# True: Dùng lại file PCA đã tính (nhanh)
# False: Tính lại PCA từ đầu (chậm, dùng cho lần đầu hoặc khi đổi data)
USE_PRECALCULATED_PCA = True

# =============================================================================
# HÀM TRÍCH XUẤT HOG MÀU (THÔ - 5292 chiều)
# =============================================================================
def extract_hog_raw(img_path):
    """Đọc ảnh màu, resize, tính HOG cho 3 kênh B-G-R và nối lại."""
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    if img_color is None: return None
    try:
        # Resize về chuẩn
        img_resized = cv2.resize(img_color, RESIZE_DIM, interpolation=cv2.INTER_AREA)
        
        # Tách kênh
        b, g, r = cv2.split(img_resized)
        
        # Tính HOG
        f_b = hog.compute(b).flatten()
        f_g = hog.compute(g).flatten()
        f_r = hog.compute(r).flatten()
        
        # Nối lại (5292 chiều)
        return np.hstack((f_b, f_g, f_r))
    except: return None

# =============================================================================
# QUY TRÌNH HUẤN LUYỆN
# =============================================================================
def train_recognizer_pca(dataset_dir, svm_model_path, pca_model_path):
    print("="*70)
    print("  TRAINING RECOGNIZER (MULTI-CLASS) VỚI COLOR HOG + PCA")
    print(f"  PCA Mode: {'LOAD CÓ SẴN' if USE_PRECALCULATED_PCA else 'TÍNH MỚI'}")
    print("="*70)

    # --- BƯỚC 1: DUYỆT DỮ LIỆU ---
    print("Đang quét thư mục dữ liệu...")
    class_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if not class_dirs:
        print(f"❌ Lỗi: Không tìm thấy thư mục lớp nào trong {dataset_dir}")
        return

    all_paths = []
    all_labels = []
    label_map = {} # Map ID -> Tên lớp (để in ra)

    for class_name in class_dirs:
        try:
            # Giả định tên thư mục là "0_stop", "1_left"...
            label_id = int(class_name.split('_')[0])
            label_map[label_id] = class_name
            
            class_path = os.path.join(dataset_dir, class_name)
            img_paths = glob.glob(os.path.join(class_path, "*.*"))
            
            for p in img_paths:
                all_paths.append(p)
                all_labels.append(label_id)
                
            print(f"  + Lớp {label_id} ({class_name}): {len(img_paths)} ảnh")
        except ValueError:
            print(f"  ⚠️ Bỏ qua thư mục sai định dạng: {class_name}")

    total_files = len(all_paths)
    print(f"✓ Tổng cộng: {total_files} ảnh từ {len(label_map)} lớp.")
    
    if total_files == 0: return

    # --- BƯỚC 2: XỬ LÝ PCA (Tính mới hoặc Load) ---
    mean = None
    eigenvectors = None
    
    # Quyết định có cần tính lại PCA không
    need_compute = not USE_PRECALCULATED_PCA
    if USE_PRECALCULATED_PCA and not os.path.exists(pca_model_path):
        print(f"⚠️ Không thấy file {pca_model_path}. Sẽ tính toán lại.")
        need_compute = True

    if need_compute:
        print("\n--- GIAI ĐOẠN 1: Tính toán ma trận PCA ---")
        # Lấy mẫu ngẫu nhiên (tối đa 20k ảnh) để tính PCA cho nhanh
        sample_size = min(20000, total_files)
        print(f"Sử dụng {sample_size} mẫu để học PCA...")
        
        # Lấy mẫu ngẫu nhiên từ danh sách đường dẫn
        sample_indices = random.sample(range(total_files), sample_size)
        pca_data = []
        
        for i, idx in enumerate(sample_indices):
            feat = extract_hog_raw(all_paths[idx])
            if feat is not None:
                pca_data.append(feat)
            if i % 1000 == 0: print(f"  Loading sample: {i}/{sample_size}", end='\r')
            
        pca_matrix = np.array(pca_data, dtype=np.float32)
        print(f"\n  Tính PCA trên ma trận: {pca_matrix.shape}...")
        
        mean, eigenvectors = cv2.PCACompute(pca_matrix, mean=None, maxComponents=PCA_COMPONENTS)
        print(f"  ✓ PCA hoàn tất. Nén xuống {eigenvectors.shape[0]} chiều.")
        
        # Lưu file PCA
        print(f"  Lưu PCA vào: {pca_model_path}")
        fs = cv2.FileStorage(pca_model_path, cv2.FILE_STORAGE_WRITE)
        fs.write("mean", mean)
        fs.write("eigenvectors", eigenvectors)
        fs.release()
        
        del pca_data, pca_matrix
        gc.collect()
    else:
        print(f"\n--- GIAI ĐOẠN 1: Tải PCA từ {pca_model_path} ---")
        fs = cv2.FileStorage(pca_model_path, cv2.FILE_STORAGE_READ)
        mean = fs.getNode("mean").mat()
        eigenvectors = fs.getNode("eigenvectors").mat()
        fs.release()
        print(f"  ✓ Đã tải xong. Số chiều: {eigenvectors.shape[0]}")

    # --- BƯỚC 3: LOAD TOÀN BỘ DATA & NÉN ---
    print("\n--- GIAI ĐOẠN 2: Trích xuất & Nén toàn bộ dữ liệu ---")
    
    actual_components = eigenvectors.shape[0]
    X_train = np.zeros((total_files, actual_components), dtype=np.float32)
    y_train = np.zeros(total_files, dtype=np.int32)
    
    valid_idx = 0
    for i, (path, label) in enumerate(zip(all_paths, all_labels)):
        feat_raw = extract_hog_raw(path)
        if feat_raw is not None:
            # Nén: (Vector - Mean) * Eigenvectors.T
            feat_pca = cv2.PCAProject(feat_raw.reshape(1, -1), mean, eigenvectors)
            
            X_train[valid_idx] = feat_pca.flatten()
            y_train[valid_idx] = label
            valid_idx += 1
            
        if i % 1000 == 0: print(f"  Processing: {i}/{total_files}", end='\r')
        
    # Cắt bỏ phần thừa
    X_train = X_train[:valid_idx]
    y_train = y_train[:valid_idx]
    
    print(f"\n  ✓ Đã chuẩn bị {valid_idx} mẫu huấn luyện.")
    print(f"  Kích thước input: {X_train.shape}")

    # --- BƯỚC 4: HUẤN LUYỆN SVM ---
    print("\n--- GIAI ĐOẠN 3: Train SVM (Multi-class) ---")
    
    # Xáo trộn
    indices = np.arange(valid_idx)
    random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print("Cấu hình SVM (RBF Kernel)...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    
    print("Bắt đầu TrainAuto (Grid Search)...")
    # kFold=5 là hợp lý cho recognizer
    svm.trainAuto(
        X_train, # Samples
        cv2.ml.ROW_SAMPLE, # Layout
        y_train, # Responses
        kFold=5
    )
    
    print("\n✓ Huấn luyện hoàn tất!")
    print(f"  -> Best C: {svm.getC()}")
    print(f"  -> Best Gamma: {svm.getGamma()}")
    
    svm.save(svm_model_path)
    print(f"✓ Đã lưu model SVM tại: {svm_model_path}")
    print("="*70)

if __name__ == "__main__":
    # INPUT: Thư mục chứa các folder con (0_xxx, 1_xxx...)
    DATASET_DIR = "../data_recognizer" 
    
    # OUTPUT: Tạo thư mục models nếu chưa có
    if not os.path.exists("models"): os.makedirs("models")
    
    # Tên file xuất ra
    # Lưu ý: Đặt tên khác với PCA của detector để tránh nhầm lẫn
    SVM_PATH = "models/svm_recognizer_color_pca_v1.xml"
    PCA_PATH = "models/pca_recognizer_v1.xml" 
    
    if os.path.exists(DATASET_DIR):
        train_recognizer_pca(DATASET_DIR, SVM_PATH, PCA_PATH)
    else:
        print(f"Lỗi: Không tìm thấy thư mục {DATASET_DIR}")