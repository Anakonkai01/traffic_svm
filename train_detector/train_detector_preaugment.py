"""
Training script for Pre-Augmented Dataset
No on-the-fly augmentation - just load pre-augmented images
Much faster and more memory-efficient!
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
class PreAugmentedDetectorConfig:
    """Configuration for training with pre-augmented data"""
    
    # Dataset paths - Point to PRE-AUGMENTED directories
    POSITIVE_SAMPLES_DIR = "dataset_augmented/positives"
    NEGATIVE_SAMPLES_DIR = "dataset_augmented/negatives"
    
    # Output
    OUTPUT_MODEL_PATH = "svm_sign_detector_v4.xml"
    
    # HOG parameters
    HOG_WIN_SIZE = (64, 64)
    HOG_BLOCK_SIZE = (16, 16)
    HOG_BLOCK_STRIDE = (8, 8)
    HOG_CELL_SIZE = (8, 8)
    HOG_NBINS = 9
    
    # Memory optimization (optional limits)
    # Set to None to use all images
    MAX_POSITIVE_SAMPLES = None  # Use all
    MAX_NEGATIVE_SAMPLES = None  # Use all
    
    # If you still have memory issues, set limits:
    # MAX_POSITIVE_SAMPLES = 100000
    # MAX_NEGATIVE_SAMPLES = 100000
    
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
        
        b, g, r = cv2.split(image)
        
        features_b = self.hog.compute(b).flatten()
        features_g = self.hog.compute(g).flatten()
        features_r = self.hog.compute(r).flatten()
        
        return np.hstack((features_b, features_g, features_r)).astype(np.float32)


# =============================================================================
# DATASET LOADER (No Augmentation!)
# =============================================================================
class PreAugmentedDatasetLoader:
    """Load pre-augmented dataset from disk"""
    
    def __init__(self, config):
        self.config = config
        self.hog_extractor = ColorHOGExtractor(config)
    
    def load_images_from_dir(self, directory: str, label: int,
                            max_samples: int = None) -> tuple:
        """Load pre-augmented images (NO augmentation needed!)"""
        
        if not os.path.exists(directory):
            print(f"⚠️  Directory not found: {directory}")
            return np.array([]), np.array([])
        
        # Get all image files
        image_files = list(Path(directory).glob("*.jpg")) + \
                     list(Path(directory).glob("*.png")) + \
                     list(Path(directory).glob("*.jpeg"))
        
        total_images = len(image_files)
        label_name = "Positive (Signs)" if label == 1 else "Negative (Background)"
        
        print(f"\n📂 Loading {label_name}")
        print(f"   Found {total_images} pre-augmented images")
        
        # Apply limit if specified
        if max_samples and total_images > max_samples:
            print(f"   Limiting to {max_samples} samples")
            import random
            image_files = random.sample(image_files, max_samples)
            total_images = max_samples
        
        features_list = []
        labels_list = []
        
        print(f"   Extracting features...")
        
        for idx, img_path in enumerate(image_files, 1):
            if idx % 1000 == 0:
                print(f"      ... {idx}/{total_images} ({idx/total_images*100:.1f}%)")
            
            # Load image (already augmented!)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            # Extract features (no augmentation!)
            features = self.hog_extractor.extract(img)
            features_list.append(features)
            labels_list.append(label)
        
        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int32)
        
        memory_mb = X.nbytes / (1024 * 1024)
        print(f"   ✅ Loaded {len(X)} samples ({memory_mb:.1f} MB)")
        
        return X, y
    
    def load_dataset(self) -> tuple:
        """Load complete pre-augmented dataset"""
        print("\n" + "=" * 70)
        print("📂 LOADING PRE-AUGMENTED DATASET")
        print("=" * 70)
        print("✅ No augmentation needed - loading pre-augmented images!")
        
        # Check if augmentation metadata exists
        metadata_path = Path(self.config.POSITIVE_SAMPLES_DIR).parent / 'augmentation_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"\n📋 Augmentation Info:")
            print(f"   Rotations: {metadata['config'].get('rotations', [])}")
            print(f"   Scales: {metadata['config'].get('scales', [])}")
            print(f"   Brightness: {metadata['config'].get('brightness', [])}")
        
        # Load positive samples
        X_pos, y_pos = self.load_images_from_dir(
            self.config.POSITIVE_SAMPLES_DIR,
            label=1,
            max_samples=self.config.MAX_POSITIVE_SAMPLES
        )
        
        # Load negative samples
        X_neg, y_neg = self.load_images_from_dir(
            self.config.NEGATIVE_SAMPLES_DIR,
            label=0,
            max_samples=self.config.MAX_NEGATIVE_SAMPLES
        )
        
        if len(X_pos) == 0 or len(X_neg) == 0:
            raise ValueError("❌ Dataset is empty!")
        
        # Combine
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([y_pos, y_neg])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print("\n" + "=" * 70)
        print("📊 DATASET SUMMARY")
        print("=" * 70)
        
        total_memory_mb = X.nbytes / (1024 * 1024)
        
        print(f"Total samples: {len(X)}")
        print(f"  - Positive: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        print(f"  - Negative: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Memory usage: {total_memory_mb:.1f} MB (~{total_memory_mb/1024:.2f} GB)")
        
        return X, y


# =============================================================================
# SVM TRAINER
# =============================================================================
class DetectorTrainer:
    """Train SVM detector"""
    
    def __init__(self, config):
        self.config = config
        self.svm = None
    
    def train(self, X_train, y_train):
        """Train SVM"""
        print("\n" + "=" * 70)
        print("🎓 TRAINING SVM DETECTOR")
        print("=" * 70)
        
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(self.config.SVM_KERNEL)
        
        print(f"\n⚙️  Configuration:")
        print(f"   Kernel: RBF")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Feature dimension: {X_train.shape[1]}")
        
        start_time = time.time()
        
        if self.config.USE_AUTO_TRAIN:
            print(f"\n🔍 AUTO-TRAINING (Grid Search)")
            print(f"   Cross-Validation: {self.config.KFOLD}-fold")
            print(f"   This may take a while...")
            
            self.svm.trainAuto(
                X_train,
                cv2.ml.ROW_SAMPLE,
                y_train,
                kFold=self.config.KFOLD
            )
            
            training_time = time.time() - start_time
            print(f"\n✅ Training completed in {training_time/60:.2f} minutes")
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
            print(f"\n✅ Training completed in {training_time:.2f}s")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION")
        print("=" * 70)
        
        _, y_pred = self.svm.predict(X_test)
        y_pred = y_pred.flatten().astype(np.int32)
        
        accuracy = np.mean(y_pred == y_test) * 100
        print(f"\n🎯 Accuracy: {accuracy:.2f}%")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Non-Sign', 'Sign'],
                                   digits=4))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Non-Sign  Sign")
        print(f"Actual Non-Sign    {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"       Sign        {cm[1,0]:5d}   {cm[1,1]:5d}")
        
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        
        print(f"\n   Precision: {precision*100:.2f}%")
        print(f"   Recall: {recall*100:.2f}%")
    
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
    print("  ✅ Reusable dataset")
    print("=" * 70)
    
    config = PreAugmentedDetectorConfig()
    
    print("\n📝 Configuration:")
    print(f"   Positive dir: {config.POSITIVE_SAMPLES_DIR}")
    print(f"   Negative dir: {config.NEGATIVE_SAMPLES_DIR}")
    print(f"   Output model: {config.OUTPUT_MODEL_PATH}")
    print(f"   Auto-training: {'Enabled' if config.USE_AUTO_TRAIN else 'Disabled'}")
    
    try:
        # Load pre-augmented dataset
        loader = PreAugmentedDatasetLoader(config)
        X, y = loader.load_dataset()
        
        # Split
        print("\n" + "=" * 70)
        print("✂️  SPLITTING DATASET")
        print("=" * 70)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SPLIT,
            random_state=config.RANDOM_SEED,
            stratify=y
        )
        
        print(f"   Training: {len(X_train)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        # Train
        trainer = DetectorTrainer(config)
        trainer.train(X_train, y_train)
        
        # Evaluate
        trainer.evaluate(X_test, y_test)
        
        # Save
        trainer.save_model()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()