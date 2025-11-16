"""
MINI-BATCH TRAINING FOR RECOGNIZER
Multi-class classification with incremental learning
Tận dụng TẤT CẢ data không bị giới hạn RAM
"""

import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import time
from tqdm import tqdm
import gc


# =============================================================================
# CONFIGURATION
# =============================================================================
class MiniBatchRecognizerConfig:
    """Configuration for mini-batch recognizer training"""
    
    # Dataset path (PRE-AUGMENTED)
    DATASET_ROOT = "dataset_recognizer_augmented"
    
    # Classes
    CLASS_DIRS = {
        0: "0_cam_di_nguoc_chieu",
        1: "1_cam_queo_trai",
        2: "2_cam_do_xe",
        3: "3_cam_dung_do_xe",
        4: "4_huong_ben_phai",
        5: "5_canh_bao_di_cham",
        6: "6_canh_bao_nguoi_qua_duong",
        7: "7_canh_bao_duong_gap_khuc"
    }
    
    # Feature cache directory
    FEATURE_CACHE_DIR = "feature_cache_recognizer"
    
    # Output
    OUTPUT_MODEL_PATH = "svm_recognizer_minibatch_v1.pkl"
    
    # HOG parameters
    HOG_WIN_SIZE = (64, 64)
    HOG_BLOCK_SIZE = (16, 16)
    HOG_BLOCK_STRIDE = (8, 8)
    HOG_CELL_SIZE = (8, 8)
    HOG_NBINS = 9
    
    # Mini-batch settings
    EXTRACT_BATCH_SIZE = 500   # Smaller batch for recognizer
    TRAIN_BATCH_SIZE = 1000
    
    # Training
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    MAX_ITER = 1000
    
    # Use ALL data
    USE_ALL_DATA = True


# =============================================================================
# COLOR HOG EXTRACTOR
# =============================================================================
class ColorHOGExtractor:
    """Extract Color HOG features"""
    
    def __init__(self, config):
        self.config = config
        self.hog = cv2.HOGDescriptor(
            config.HOG_WIN_SIZE,
            config.HOG_BLOCK_SIZE,
            config.HOG_BLOCK_STRIDE,
            config.HOG_CELL_SIZE,
            config.HOG_NBINS
        )
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract Color HOG"""
        if image.shape[:2] != self.config.HOG_WIN_SIZE:
            image = cv2.resize(image, self.config.HOG_WIN_SIZE,
                              interpolation=cv2.INTER_AREA)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        b, g, r = cv2.split(image)
        
        features_b = self.hog.compute(b).flatten()
        features_g = self.hog.compute(g).flatten()
        features_r = self.hog.compute(r).flatten()
        
        return np.hstack((features_b, features_g, features_r)).astype(np.float32)


# =============================================================================
# FEATURE CACHE MANAGER
# =============================================================================
class FeatureCacheManager:
    """Manage feature extraction and caching for multi-class"""
    
    def __init__(self, config):
        self.config = config
        self.hog_extractor = ColorHOGExtractor(config)
        
        os.makedirs(config.FEATURE_CACHE_DIR, exist_ok=True)
    
    def extract_and_cache_class(self, class_id: int, class_name: str):
        """Extract features for a single class"""
        
        class_dir = os.path.join(self.config.DATASET_ROOT, class_name)
        
        if not os.path.exists(class_dir):
            print(f"   ⚠️  Not found: {class_dir}")
            return 0
        
        print(f"\n📁 Class {class_id}: {class_name}")
        
        # Get all images
        image_files = list(Path(class_dir).glob("*.jpg")) + \
                     list(Path(class_dir).glob("*.png")) + \
                     list(Path(class_dir).glob("*.jpeg"))
        
        total_images = len(image_files)
        print(f"   Found {total_images} images")
        
        if total_images == 0:
            return 0
        
        # Process in batches
        batch_num = 0
        total_cached = 0
        
        for i in range(0, total_images, self.config.EXTRACT_BATCH_SIZE):
            batch_files = image_files[i:i + self.config.EXTRACT_BATCH_SIZE]
            
            features_list = []
            labels_list = []
            
            for img_path in tqdm(batch_files, desc=f"   Batch {batch_num}"):
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                features = self.hog_extractor.extract(img)
                features_list.append(features)
                labels_list.append(class_id)
            
            # Save batch
            if features_list:
                cache_file = os.path.join(
                    self.config.FEATURE_CACHE_DIR,
                    f"class_{class_id}_batch_{batch_num}.npz"
                )
                
                np.savez_compressed(
                    cache_file,
                    features=np.array(features_list, dtype=np.float32),
                    labels=np.array(labels_list, dtype=np.int32)
                )
                
                cached_count = len(features_list)
                total_cached += cached_count
                
                print(f"   ✅ Cached {cached_count} samples → {cache_file}")
            
            del features_list, labels_list
            gc.collect()
            
            batch_num += 1
        
        print(f"   Total: {total_cached} samples")
        return batch_num
    
    def extract_all_features(self):
        """Extract features for all classes"""
        
        print("\n" + "=" * 70)
        print("🔍 EXTRACTING FEATURES (MINI-BATCH MODE)")
        print("=" * 70)
        print("✅ No RAM limits - processing ALL data!")
        
        start_time = time.time()
        
        total_batches = 0
        
        for class_id, class_name in self.config.CLASS_DIRS.items():
            batches = self.extract_and_cache_class(class_id, class_name)
            total_batches += batches
        
        extraction_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("✅ FEATURE EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"   Time: {extraction_time:.2f}s ({extraction_time/60:.2f} min)")
        print(f"   Total batches: {total_batches}")
        print(f"   Cache directory: {self.config.FEATURE_CACHE_DIR}")
    
    def load_batch(self, batch_file: str):
        """Load a batch"""
        data = np.load(batch_file)
        return data['features'], data['labels']
    
    def get_all_cache_files(self):
        """Get all cache files"""
        cache_files = list(Path(self.config.FEATURE_CACHE_DIR).glob("*.npz"))
        return sorted(cache_files)
    
    def load_all_for_split(self):
        """Load all features"""
        
        print("\n📊 Loading all features for train/test split...")
        
        cache_files = self.get_all_cache_files()
        
        all_features = []
        all_labels = []
        
        for cache_file in tqdm(cache_files, desc="Loading"):
            features, labels = self.load_batch(str(cache_file))
            all_features.append(features)
            all_labels.append(labels)
        
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        print(f"   Total samples: {len(X)}")
        
        # Per-class count
        for class_id, class_name in self.config.CLASS_DIRS.items():
            count = np.sum(y == class_id)
            print(f"   [{class_id}] {class_name}: {count}")
        
        return X, y


# =============================================================================
# MINI-BATCH TRAINER
# =============================================================================
class MiniBatchTrainer:
    """Train multi-class SVM using mini-batch"""
    
    def __init__(self, config):
        self.config = config
        
        # Multi-class SVM
        self.model = SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=0.0001,
            max_iter=config.MAX_ITER,
            tol=1e-3,
            random_state=config.RANDOM_SEED,
            warm_start=True,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train model"""
        
        print("\n" + "=" * 70)
        print("🎓 TRAINING (MINI-BATCH MODE)")
        print("=" * 70)
        
        print(f"\n⚙️  Configuration:")
        print(f"   Model: Multi-class SVM (SGDClassifier)")
        print(f"   Classes: {len(self.config.CLASS_DIRS)}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Batch size: {self.config.TRAIN_BATCH_SIZE}")
        
        start_time = time.time()
        
        # Get all unique classes
        classes = np.arange(len(self.config.CLASS_DIRS))
        
        # Train in batches over multiple epochs
        print(f"\n⏳ Training...")
        
        for epoch in range(3):
            print(f"\n   Epoch {epoch + 1}/3")
            
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in tqdm(range(0, len(X_train), self.config.TRAIN_BATCH_SIZE),
                         desc="   Batches"):
                batch_X = X_shuffled[i:i+self.config.TRAIN_BATCH_SIZE]
                batch_y = y_shuffled[i:i+self.config.TRAIN_BATCH_SIZE]
                
                self.model.partial_fit(batch_X, batch_y, classes=classes)
                
                # Clear memory periodically
                if i % (self.config.TRAIN_BATCH_SIZE * 10) == 0:
                    gc.collect()
        
        training_time = time.time() - start_time
        
        print(f"\n✅ Training complete!")
        print(f"   Time: {training_time:.2f}s ({training_time/60:.2f} min)")
        print(f"   Epochs: 3")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model"""
        
        print("\n" + "=" * 70)
        print("📊 EVALUATION")
        print("=" * 70)
        
        # Predict
        if len(X_test) > 5000:
            print("   Large test set - predicting in batches...")
            y_pred = []
            
            for i in range(0, len(X_test), 2000):
                batch = X_test[i:i+2000]
                pred = self.model.predict(batch)
                y_pred.extend(pred)
            
            y_pred = np.array(y_pred)
        else:
            y_pred = self.model.predict(X_test)
        
        # Metrics
        accuracy = np.mean(y_pred == y_test) * 100
        
        print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
        
        # Classification report
        target_names = [self.config.CLASS_DIRS[i]
                       for i in sorted(self.config.CLASS_DIRS.keys())]
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=target_names,
                                   digits=4))
        
        # Per-class accuracy
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n📈 Per-Class Accuracy:")
        class_accuracies = []
        
        for i, class_name in enumerate(target_names):
            if np.sum(cm[i, :]) > 0:
                class_acc = cm[i, i] / np.sum(cm[i, :]) * 100
                class_accuracies.append(class_acc)
                status = "✅" if class_acc >= 85 else "⚠️"
                print(f"   {status} [{i}] {class_name:30s}: {class_acc:.2f}%")
        
        min_class_acc = min(class_accuracies) if class_accuracies else 0
        
        print(f"\n📊 Summary:")
        print(f"   Overall: {accuracy:.2f}%")
        print(f"   Lowest class: {min_class_acc:.2f}%")
        
        return accuracy
    
    def save_model(self):
        """Save model"""
        
        print("\n" + "=" * 70)
        print("💾 SAVING MODEL")
        print("=" * 70)
        
        with open(self.config.OUTPUT_MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✅ Model saved: {self.config.OUTPUT_MODEL_PATH}")
        print(f"   Size: {os.path.getsize(self.config.OUTPUT_MODEL_PATH) / (1024**2):.2f} MB")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main pipeline"""
    
    print("\n" + "=" * 70)
    print("🚀 MINI-BATCH RECOGNIZER TRAINING")
    print("=" * 70)
    print("Strategy: Extract → Cache → Train incrementally")
    print("=" * 70)
    
    config = MiniBatchRecognizerConfig()
    
    print("\n📝 Configuration:")
    print(f"   Dataset: {config.DATASET_ROOT}")
    print(f"   Classes: {len(config.CLASS_DIRS)}")
    print(f"   Cache: {config.FEATURE_CACHE_DIR}")
    print(f"   Use all data: {config.USE_ALL_DATA}")
    
    try:
        # Extract features
        cache_manager = FeatureCacheManager(config)
        
        cache_files = cache_manager.get_all_cache_files()
        
        if cache_files:
            print(f"\n⚠️  Found {len(cache_files)} cached files")
            response = input("   Use existing cache? (y/n): ")
            
            if response.lower() != 'y':
                cache_manager.extract_all_features()
        else:
            cache_manager.extract_all_features()
        
        # Load all for split
        X, y = cache_manager.load_all_for_split()
        
        # Split
        print("\n" + "=" * 70)
        print("✂️  SPLITTING")
        print("=" * 70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SPLIT,
            random_state=config.RANDOM_SEED,
            stratify=y
        )
        
        print(f"   Training: {len(X_train)}")
        print(f"   Test: {len(X_test)}")
        
        # Train
        trainer = MiniBatchTrainer(config)
        trainer.train(X_train, y_train)
        
        # Evaluate
        trainer.evaluate(X_test, y_test)
        
        # Save
        trainer.save_model()
        
        print("\n" + "=" * 70)
        print("✅ COMPLETED!")
        print("=" * 70)
        print(f"📁 Model: {config.OUTPUT_MODEL_PATH}")
        print(f"📁 Cache: {config.FEATURE_CACHE_DIR}/")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()