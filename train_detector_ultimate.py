"""
ULTIMATE Training script for SVM Detector (Model B)
Combines best features from both versions:
- Color HOG (from user's file)
- trainAuto() + Grid Search (from user's file)
- Data Augmentation (from my file)
- Train/Test Split & Evaluation (from my file)
- OOP Structure (from my file)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time


# =============================================================================
# CONFIGURATION
# =============================================================================
class UltimateDetectorConfig:
    """Ultimate configuration for detector training"""
    
    # Dataset paths
    POSITIVE_SAMPLES_DIR = "data_detector/positives"  # Chứa ảnh có biển báo
    NEGATIVE_SAMPLES_DIR = "data_detector/negatives"  # Chứa ảnh không có biển báo

    # Output
    OUTPUT_MODEL_PATH = "svm_sign_detector_hog_color_v2.xml"
    
    # HOG parameters - MUST MATCH detection pipeline
    HOG_WIN_SIZE = (64, 64)
    HOG_BLOCK_SIZE = (16, 16)
    HOG_BLOCK_STRIDE = (8, 8)
    HOG_CELL_SIZE = (8, 8)
    HOG_NBINS = 9
    
    # Data augmentation
    USE_AUGMENTATION = False
    AUGMENT_ROTATIONS = [-15, -10, -5, 5, 10, 15]  # Degrees
    AUGMENT_SCALES = [0.9, 0.95, 1.05, 1.1]
    AUGMENT_BRIGHTNESS = [-30, -15, 15, 30]
    
    # Training parameters
    TEST_SPLIT = 0.2  # 20% for testing
    RANDOM_SEED = 42
    
    # SVM parameters - Will use trainAuto() for optimization
    SVM_KERNEL = cv2.ml.SVM_RBF  # RBF kernel for best accuracy
    USE_AUTO_TRAIN = True  # Use Grid Search to find best C and Gamma
    KFOLD = 3  # Cross-validation folds
    
    # Manual params (only used if USE_AUTO_TRAIN = False)
    SVM_C = 10.0
    SVM_GAMMA = 0.01


# =============================================================================
# COLOR HOG EXTRACTOR (User's method - Best for color signs)
# =============================================================================
class ColorHOGExtractor:
    """Extract HOG features from color channels (B, G, R)"""
    
    def __init__(self, config: UltimateDetectorConfig):
        self.config = config
        self.hog = cv2.HOGDescriptor(
            config.HOG_WIN_SIZE,
            config.HOG_BLOCK_SIZE,
            config.HOG_BLOCK_STRIDE,
            config.HOG_CELL_SIZE,
            config.HOG_NBINS
        )
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract Color HOG features (B, G, R channels separately)"""
        # Resize to standard size
        if image.shape[:2] != self.config.HOG_WIN_SIZE:
            image = cv2.resize(image, self.config.HOG_WIN_SIZE, 
                              interpolation=cv2.INTER_AREA)
        
        # Split into B, G, R channels
        b_channel, g_channel, r_channel = cv2.split(image)
        
        # Extract HOG from each channel
        features_b = self.hog.compute(b_channel).flatten()
        features_g = self.hog.compute(g_channel).flatten()
        features_r = self.hog.compute(r_channel).flatten()
        
        # Concatenate all channels
        color_hog = np.hstack((features_b, features_g, features_r))
        
        return color_hog.astype(np.float32)


# =============================================================================
# DATA AUGMENTATION
# =============================================================================
class DataAugmenter:
    """Data augmentation for training"""
    
    def __init__(self, config: UltimateDetectorConfig):
        self.config = config
    
    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle (degrees)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def scale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale image"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(image, (new_w, new_h))
        
        # Crop or pad to original size
        if scale_factor > 1.0:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            return scaled[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            return cv2.copyMakeBorder(scaled, pad_y, h-new_h-pad_y, 
                                     pad_x, w-new_w-pad_x, 
                                     cv2.BORDER_REPLICATE)
    
    def adjust_brightness(self, image: np.ndarray, value: int) -> np.ndarray:
        """Adjust brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255).astype(np.uint8)
        
        final_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    def augment(self, image: np.ndarray) -> list:
        """Generate augmented versions of image"""
        augmented = [image.copy()]  # Original
        
        if not self.config.USE_AUGMENTATION:
            return augmented
        
        # Rotations
        for angle in self.config.AUGMENT_ROTATIONS:
            augmented.append(self.rotate(image, angle))
        
        # Scales
        for scale in self.config.AUGMENT_SCALES:
            augmented.append(self.scale(image, scale))
        
        # Brightness
        for brightness in self.config.AUGMENT_BRIGHTNESS:
            augmented.append(self.adjust_brightness(image, brightness))
        
        return augmented


# =============================================================================
# DATASET LOADER
# =============================================================================
class DetectorDatasetLoader:
    """Load and prepare dataset for detector training"""
    
    def __init__(self, config: UltimateDetectorConfig):
        self.config = config
        self.hog_extractor = ColorHOGExtractor(config)
        self.augmenter = DataAugmenter(config)
    
    def load_images_from_dir(self, directory: str, label: int, 
                            use_augmentation: bool = True) -> tuple:
        """Load images from directory and extract features"""
        features_list = []
        labels_list = []
        
        if not os.path.exists(directory):
            print(f"⚠️  Directory not found: {directory}")
            return np.array([]), np.array([])
        
        image_files = list(Path(directory).glob("*.jpg")) + \
                     list(Path(directory).glob("*.png")) + \
                     list(Path(directory).glob("*.jpeg"))
        
        label_name = "Positive (Signs)" if label == 1 else "Negative (Background)"
        print(f"   Loading {len(image_files)} images from {directory}...")
        
        for idx, img_path in enumerate(image_files, 1):
            if idx % 100 == 0:
                print(f"      ... Processed {idx} / {len(image_files)} images...")
            
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            # Augment
            if use_augmentation and self.config.USE_AUGMENTATION:
                augmented_images = self.augmenter.augment(img)
            else:
                augmented_images = [img]
            
            # Extract features for each augmented version
            for aug_img in augmented_images:
                features = self.hog_extractor.extract(aug_img)
                features_list.append(features)
                labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
    
    def load_dataset(self) -> tuple:
        """Load complete dataset"""
        print("\n" + "=" * 70)
        print("📂 LOADING DATASET (Color HOG)")
        print("=" * 70)
        
        # Load positive samples (signs)
        print("\n1️⃣  Positive samples (traffic signs):")
        X_pos, y_pos = self.load_images_from_dir(
            self.config.POSITIVE_SAMPLES_DIR, 
            label=1,  # Positive class
            use_augmentation=True
        )
        print(f"   ✅ Loaded {len(X_pos)} positive samples (with augmentation)")
        
        # Load negative samples (non-signs)
        print("\n2️⃣  Negative samples (background/non-signs):")
        X_neg, y_neg = self.load_images_from_dir(
            self.config.NEGATIVE_SAMPLES_DIR,
            label=0,  # Negative class
            use_augmentation=True
        )
        print(f"   ✅ Loaded {len(X_neg)} negative samples (with augmentation)")
        
        # Combine
        if len(X_pos) == 0 or len(X_neg) == 0:
            raise ValueError("❌ Dataset is empty! Check your directories.")
        
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([y_pos, y_neg])
        
        print("\n📊 Dataset Summary:")
        print(f"   Total samples: {len(X)}")
        print(f"   Positive: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
        print(f"   Negative: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"   Feature dimension: {X.shape[1]} (Color HOG: B+G+R)")
        
        return X, y


# =============================================================================
# ULTIMATE SVM TRAINER
# =============================================================================
class UltimateDetectorTrainer:
    """Train SVM with Grid Search optimization"""
    
    def __init__(self, config: UltimateDetectorConfig):
        self.config = config
        self.svm = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train SVM model with optimal hyperparameters"""
        print("\n" + "=" * 70)
        print("🎓 TRAINING SVM DETECTOR (ULTIMATE VERSION)")
        print("=" * 70)
        
        # Create SVM
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(self.config.SVM_KERNEL)
        
        print(f"\n⚙️  SVM Configuration:")
        print(f"   Kernel: RBF (Radial Basis Function)")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Feature dimension: {X_train.shape[1]}")
        
        # Train
        start_time = time.time()
        
        if self.config.USE_AUTO_TRAIN:
            print(f"\n🔍 AUTO-TRAINING MODE (Grid Search)")
            print(f"   Cross-Validation: {self.config.KFOLD}-fold")
            print(f"   ⚠️  This will take A LONG TIME (possibly hours)")
            print(f"   ⚠️  OpenCV doesn't show progress - please be patient...")
            print(f"   ⏳ Starting Grid Search...")
            
            # Use trainAuto for automatic hyperparameter optimization
            self.svm.trainAuto(
                X_train, 
                cv2.ml.ROW_SAMPLE, 
                y_train.astype(np.int32),
                kFold=self.config.KFOLD
            )
            
            training_time = time.time() - start_time
            
            print(f"\n✅ Auto-Training completed in {training_time/60:.2f} minutes!")
            print(f"\n📊 Optimal Hyperparameters Found:")
            print(f"   Best C: {self.svm.getC():.6f}")
            print(f"   Best Gamma: {self.svm.getGamma():.6f}")
            
        else:
            print(f"\n⚡ MANUAL TRAINING MODE")
            print(f"   C: {self.config.SVM_C}")
            print(f"   Gamma: {self.config.SVM_GAMMA}")
            
            self.svm.setC(self.config.SVM_C)
            self.svm.setGamma(self.config.SVM_GAMMA)
            self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 
                                     1000, 1e-6))
            
            print(f"\n⏳ Training in progress...")
            self.svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32))
            
            training_time = time.time() - start_time
            print(f"✅ Training completed in {training_time:.2f}s")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Evaluate model on test set"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION ON TEST SET")
        print("=" * 70)
        
        # Predict
        _, y_pred = self.svm.predict(X_test)
        y_pred = y_pred.flatten().astype(np.int32)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test) * 100
        
        print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Non-Sign (0)', 'Sign (1)'],
                                   digits=4))
        
        # Confusion matrix
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                 Predicted")
        print(f"                 Non-Sign  Sign")
        print(f"Actual Non-Sign    {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"       Sign        {cm[1,0]:5d}   {cm[1,1]:5d}")
        
        # Calculate precision and recall for positive class
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📈 Metrics for Sign Detection (Class 1):")
        print(f"   Precision: {precision*100:.2f}% (trong số dự đoán là Sign, bao nhiêu % đúng)")
        print(f"   Recall:    {recall*100:.2f}% (trong số Sign thật, bao nhiêu % phát hiện được)")
        print(f"   F1-Score:  {f1*100:.2f}%")
        
        # Evaluation
        if accuracy >= 95 and recall >= 95 and precision >= 90:
            print(f"\n✅✅✅ EXCELLENT! Model meets all target metrics!")
        elif accuracy >= 90:
            print(f"\n✅ GOOD! Model performs well.")
        else:
            print(f"\n⚠️  Model needs improvement. Consider:")
            print(f"      - Collecting more training data")
            print(f"      - Adjusting hyperparameters")
            print(f"      - Checking data quality")
    
    def save_model(self) -> None:
        """Save trained model"""
        print("\n" + "=" * 70)
        print("💾 SAVING MODEL")
        print("=" * 70)
        
        self.svm.save(self.config.OUTPUT_MODEL_PATH)
        print(f"✅ Model saved to: {self.config.OUTPUT_MODEL_PATH}")
        print(f"\n📝 Model Details:")
        print(f"   Features: Color HOG (B+G+R)")
        print(f"   Kernel: RBF")
        print(f"   C: {self.svm.getC():.6f}")
        print(f"   Gamma: {self.svm.getGamma():.6f}")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("🚀 ULTIMATE TRAFFIC SIGN DETECTOR TRAINING")
    print("=" * 70)
    print("Features:")
    print("  ✅ Color HOG (B, G, R channels)")
    print("  ✅ Grid Search + Auto-Training")
    print("  ✅ Data Augmentation")
    print("  ✅ Train/Test Split & Evaluation")
    print("=" * 70)
    
    # Initialize configuration
    config = UltimateDetectorConfig()
    
    # Print configuration
    print("\n📝 Configuration:")
    print(f"   Positive samples: {config.POSITIVE_SAMPLES_DIR}")
    print(f"   Negative samples: {config.NEGATIVE_SAMPLES_DIR}")
    print(f"   Output model: {config.OUTPUT_MODEL_PATH}")
    print(f"   Test split: {config.TEST_SPLIT * 100:.0f}%")
    print(f"   Augmentation: {'✅ Enabled' if config.USE_AUGMENTATION else '❌ Disabled'}")
    print(f"   Auto-Training: {'✅ Enabled (Grid Search)' if config.USE_AUTO_TRAIN else '❌ Disabled (Manual)'}")
    
    try:
        # Load dataset
        loader = DetectorDatasetLoader(config)
        X, y = loader.load_dataset()
        
        # Split train/test
        print("\n" + "=" * 70)
        print("✂️  SPLITTING DATASET")
        print("=" * 70)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SPLIT, 
            random_state=config.RANDOM_SEED,
            stratify=y
        )
        
        print(f"   Training set: {len(X_train)} samples")
        print(f"      - Positive: {np.sum(y_train == 1)}")
        print(f"      - Negative: {np.sum(y_train == 0)}")
        print(f"   Test set: {len(X_test)} samples")
        print(f"      - Positive: {np.sum(y_test == 1)}")
        print(f"      - Negative: {np.sum(y_test == 0)}")
        
        # Train model
        trainer = UltimateDetectorTrainer(config)
        trainer.train(X_train, y_train)
        
        # Evaluate model
        trainer.evaluate(X_test, y_test)
        
        # Save model
        trainer.save_model()
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 Model saved: {config.OUTPUT_MODEL_PATH}")
        print("🎯 Next step: Train recognizer using train_recognizer_ultimate.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()