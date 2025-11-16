"""
Recognizer Training with Pre-Augmented Dataset
Load pre-augmented images - no on-the-fly augmentation needed!
"""

import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import json


# =============================================================================
# CONFIGURATION
# =============================================================================
class PreAugmentedRecognizerConfig:
    """Configuration for pre-augmented recognizer training"""
    
    # Dataset path - Point to PRE-AUGMENTED directory
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
    
    # Output
    OUTPUT_MODEL_PATH = "svm_sign_recognizer_v3.xml"
    
    # HOG parameters
    HOG_WIN_SIZE = (64, 64)
    HOG_BLOCK_SIZE = (16, 16)
    HOG_BLOCK_STRIDE = (8, 8)
    HOG_CELL_SIZE = (8, 8)
    HOG_NBINS = 9
    
    # Memory optimization (optional)
    MAX_SAMPLES_PER_CLASS = None  # Use all
    # If memory issues: MAX_SAMPLES_PER_CLASS = 20000
    
    # Training parameters
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # SVM parameters
    SVM_KERNEL = cv2.ml.SVM_RBF
    USE_AUTO_TRAIN = True
    KFOLD = 3
    
    SVM_C = 10.0
    SVM_GAMMA = 0.01


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
# DATASET LOADER
# =============================================================================
class PreAugmentedDatasetLoader:
    """Load pre-augmented recognizer dataset"""
    
    def __init__(self, config):
        self.config = config
        self.hog_extractor = ColorHOGExtractor(config)
    
    def load_class_images(self, class_id: int, class_name: str) -> tuple:
        """Load pre-augmented images for a class"""
        class_dir = os.path.join(self.config.DATASET_ROOT, class_name)
        
        if not os.path.exists(class_dir):
            print(f"   ⚠️  Not found: {class_dir}")
            return np.array([]), np.array([])
        
        image_files = list(Path(class_dir).glob("*.jpg")) + \
                     list(Path(class_dir).glob("*.png")) + \
                     list(Path(class_dir).glob("*.jpeg"))
        
        total_images = len(image_files)
        
        # Apply limit if needed
        if self.config.MAX_SAMPLES_PER_CLASS and total_images > self.config.MAX_SAMPLES_PER_CLASS:
            print(f"   Limiting to {self.config.MAX_SAMPLES_PER_CLASS} samples")
            import random
            image_files = random.sample(image_files, self.config.MAX_SAMPLES_PER_CLASS)
            total_images = self.config.MAX_SAMPLES_PER_CLASS
        
        print(f"   Loading {total_images} pre-augmented images...")
        
        features_list = []
        labels_list = []
        
        for idx, img_path in enumerate(image_files, 1):
            if idx % 500 == 0:
                print(f"      ... {idx}/{total_images}")
            
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            # Extract features (no augmentation!)
            features = self.hog_extractor.extract(img)
            features_list.append(features)
            labels_list.append(class_id)
        
        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int32)
        
        memory_mb = X.nbytes / (1024 * 1024)
        print(f"   ✅ Loaded {len(X)} samples ({memory_mb:.1f} MB)")
        
        return X, y
    
    def load_dataset(self) -> tuple:
        """Load complete dataset"""
        print("\n" + "=" * 70)
        print("📂 LOADING PRE-AUGMENTED DATASET")
        print("=" * 70)
        print("✅ No augmentation needed!")
        
        # Check metadata
        metadata_path = Path(self.config.DATASET_ROOT) / 'augmentation_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"\n📋 Augmentation Info:")
            print(f"   Rotations: {metadata['config'].get('rotations', [])}")
            print(f"   Scales: {metadata['config'].get('scales', [])}")
        
        all_features = []
        all_labels = []
        class_counts = {}
        
        for class_id, class_name in self.config.CLASS_DIRS.items():
            print(f"\n📁 Class {class_id}: {class_name}")
            X_class, y_class = self.load_class_images(class_id, class_name)
            
            if len(X_class) > 0:
                all_features.append(X_class)
                all_labels.append(y_class)
                class_counts[class_name] = len(X_class)
        
        if not all_features:
            raise ValueError("❌ Dataset is empty!")
        
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print("\n" + "=" * 70)
        print("📊 DATASET SUMMARY")
        print("=" * 70)
        
        total_memory_mb = X.nbytes / (1024 * 1024)
        print(f"Total samples: {len(X)}")
        print(f"Memory usage: {total_memory_mb:.1f} MB (~{total_memory_mb/1024:.2f} GB)")
        
        print(f"\nSamples per class:")
        for class_id, class_name in self.config.CLASS_DIRS.items():
            count = class_counts.get(class_name, 0)
            percentage = count / len(X) * 100 if len(X) > 0 else 0
            print(f"   [{class_id}] {class_name:30s}: {count:5d} ({percentage:.1f}%)")
        
        return X, y


# =============================================================================
# TRAINER
# =============================================================================
class RecognizerTrainer:
    """Train SVM recognizer"""
    
    def __init__(self, config):
        self.config = config
        self.svm = None
    
    def train(self, X_train, y_train):
        """Train SVM"""
        print("\n" + "=" * 70)
        print("🎓 TRAINING SVM RECOGNIZER")
        print("=" * 70)
        
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(self.config.SVM_KERNEL)
        
        print(f"\n⚙️  Configuration:")
        print(f"   Kernel: RBF")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Classes: {len(self.config.CLASS_DIRS)}")
        
        start_time = time.time()
        
        if self.config.USE_AUTO_TRAIN:
            print(f"\n🔍 AUTO-TRAINING")
            print(f"   Cross-Validation: {self.config.KFOLD}-fold")
            
            self.svm.trainAuto(
                X_train,
                cv2.ml.ROW_SAMPLE,
                y_train,
                kFold=self.config.KFOLD
            )
            
            training_time = time.time() - start_time
            print(f"\n✅ Completed in {training_time/60:.2f} minutes")
            print(f"   Best C: {self.svm.getC():.6f}")
            print(f"   Best Gamma: {self.svm.getGamma():.6f}")
        else:
            print(f"\n⚡ MANUAL TRAINING")
            self.svm.setC(self.config.SVM_C)
            self.svm.setGamma(self.config.SVM_GAMMA)
            self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER +
                                     cv2.TERM_CRITERIA_EPS, 1000, 1e-6))
            
            self.svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
            
            training_time = time.time() - start_time
            print(f"\n✅ Completed in {training_time:.2f}s")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION")
        print("=" * 70)
        
        _, y_pred = self.svm.predict(X_test)
        y_pred = y_pred.flatten().astype(np.int32)
        
        accuracy = np.mean(y_pred == y_test) * 100
        print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
        
        target_names = [self.config.CLASS_DIRS[i]
                       for i in sorted(self.config.CLASS_DIRS.keys())]
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=target_names, digits=4))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\n📈 Per-Class Accuracy:")
        for i, class_name in enumerate(target_names):
            if np.sum(cm[i, :]) > 0:
                class_acc = cm[i, i] / np.sum(cm[i, :]) * 100
                print(f"   [{i}] {class_name:30s}: {class_acc:.2f}%")
    
    def save_model(self):
        """Save model"""
        print("\n" + "=" * 70)
        print("💾 SAVING MODEL")
        print("=" * 70)
        
        self.svm.save(self.config.OUTPUT_MODEL_PATH)
        print(f"✅ Model saved: {self.config.OUTPUT_MODEL_PATH}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("🚀 TRAINING WITH PRE-AUGMENTED DATASET")
    print("=" * 70)
    print("Benefits:")
    print("  ✅ No augmentation overhead")
    print("  ✅ Faster training")
    print("  ✅ More memory-efficient")
    print("=" * 70)
    
    config = PreAugmentedRecognizerConfig()
    
    print("\n📝 Configuration:")
    print(f"   Dataset: {config.DATASET_ROOT}")
    print(f"   Classes: {len(config.CLASS_DIRS)}")
    print(f"   Auto-training: {'Enabled' if config.USE_AUTO_TRAIN else 'Disabled'}")
    
    try:
        # Load
        loader = PreAugmentedDatasetLoader(config)
        X, y = loader.load_dataset()
        
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
        trainer = RecognizerTrainer(config)
        trainer.train(X_train, y_train)
        
        # Evaluate
        trainer.evaluate(X_test, y_test)
        
        # Save
        trainer.save_model()
        
        print("\n" + "=" * 70)
        print("✅ COMPLETED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()