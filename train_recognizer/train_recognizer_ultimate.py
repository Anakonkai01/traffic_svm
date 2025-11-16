"""
ULTIMATE Training script for SVM Recognizer (Model A)
Combines best features from both versions:
- Color HOG (from user's file - changed from my grayscale version)
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
class UltimateRecognizerConfig:
    """Ultimate configuration for recognizer training"""
    
    # Dataset path - organized by class
    DATASET_ROOT = "data_recognizer"
    
    # Class directories (must match your label names)
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
    OUTPUT_MODEL_PATH = "svm_sign_recognizer_hog_color_v2.xml"
    
    # HOG parameters - MUST MATCH detection pipeline
    HOG_WIN_SIZE = (64, 64)
    HOG_BLOCK_SIZE = (16, 16)
    HOG_BLOCK_STRIDE = (8, 8)
    HOG_CELL_SIZE = (8, 8)
    HOG_NBINS = 9
    
    # Data augmentation
    USE_AUGMENTATION = False
    AUGMENT_ROTATIONS = [-15, -10, -5, 5, 10, 15]
    AUGMENT_SCALES = [0.9, 0.95, 1.05, 1.1]
    AUGMENT_BRIGHTNESS = [-30, -15, 15, 30]
    AUGMENT_FLIPS = True  # Flip horizontal (for some signs)
    
    # Training parameters
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # SVM parameters - Will use trainAuto() for optimization
    SVM_KERNEL = cv2.ml.SVM_RBF
    USE_AUTO_TRAIN = True  # Use Grid Search
    KFOLD = 5
    
    # Manual params (only used if USE_AUTO_TRAIN = False)
    SVM_C = 10.0
    SVM_GAMMA = 0.01


# =============================================================================
# COLOR HOG EXTRACTOR (User's method - Best for color signs)
# =============================================================================
class ColorHOGExtractor:
    """Extract HOG features from color channels (B, G, R)"""
    
    def __init__(self, config: UltimateRecognizerConfig):
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
        
        # Ensure color image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
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
    
    def __init__(self, config: UltimateRecognizerConfig):
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
        
        if scale_factor > 1.0:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            return scaled[start_y:start_y+h, start_x:start_x+w]
        else:
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
    
    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally"""
        return cv2.flip(image, 1)
    
    def augment(self, image: np.ndarray, allow_flip: bool = True) -> list:
        """Generate augmented versions of image"""
        augmented = [image.copy()]
        
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
        
        # Flip (only for symmetric signs)
        if allow_flip and self.config.AUGMENT_FLIPS:
            augmented.append(self.flip_horizontal(image))
        
        return augmented


# =============================================================================
# DATASET LOADER
# =============================================================================
class RecognizerDatasetLoader:
    """Load and prepare dataset for recognizer training"""
    
    def __init__(self, config: UltimateRecognizerConfig):
        self.config = config
        self.hog_extractor = ColorHOGExtractor(config)
        self.augmenter = DataAugmenter(config)
    
    def load_class_images(self, class_id: int, class_name: str) -> tuple:
        """Load images for a specific class"""
        class_dir = os.path.join(self.config.DATASET_ROOT, class_name)
        
        if not os.path.exists(class_dir):
            print(f"   ⚠️  Directory not found: {class_dir}")
            return np.array([]), np.array([])
        
        image_files = list(Path(class_dir).glob("*.jpg")) + \
                     list(Path(class_dir).glob("*.png")) + \
                     list(Path(class_dir).glob("*.jpeg"))
        
        features_list = []
        labels_list = []
        
        # Determine if this sign should be flipped
        # Don't flip directional signs
        no_flip_signs = ["cam_queo_trai", "huong_ben_phai"]
        allow_flip = class_name not in no_flip_signs
        
        for idx, img_path in enumerate(image_files, 1):
            if idx % 50 == 0:
                print(f"      ... Processed {idx} / {len(image_files)} images...")
            
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            # Augment
            if self.config.USE_AUGMENTATION:
                augmented_images = self.augmenter.augment(img, allow_flip)
            else:
                augmented_images = [img]
            
            # Extract features
            for aug_img in augmented_images:
                features = self.hog_extractor.extract(aug_img)
                features_list.append(features)
                labels_list.append(class_id)
        
        return np.array(features_list), np.array(labels_list)
    
    def load_dataset(self) -> tuple:
        """Load complete dataset"""
        print("\n" + "=" * 70)
        print("📂 LOADING DATASET (Color HOG)")
        print("=" * 70)
        
        all_features = []
        all_labels = []
        class_counts = {}
        
        # Load each class
        for class_id, class_name in self.config.CLASS_DIRS.items():
            print(f"\n📁 Class {class_id}: {class_name}")
            X_class, y_class = self.load_class_images(class_id, class_name)
            
            if len(X_class) > 0:
                all_features.append(X_class)
                all_labels.append(y_class)
                class_counts[class_name] = len(X_class)
                print(f"   ✅ Loaded {len(X_class)} samples (with augmentation)")
            else:
                print(f"   ⚠️  No samples found!")
        
        if not all_features:
            raise ValueError("❌ Dataset is empty! Check your directories.")
        
        # Combine all classes
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 DATASET SUMMARY")
        print("=" * 70)
        print(f"Total samples: {len(X)}")
        print(f"Feature dimension: {X.shape[1]} (Color HOG: B+G+R)")
        print(f"\nSamples per class:")
        for class_id, class_name in self.config.CLASS_DIRS.items():
            count = class_counts.get(class_name, 0)
            percentage = count / len(X) * 100 if len(X) > 0 else 0
            print(f"   [{class_id}] {class_name:30s}: {count:5d} ({percentage:.1f}%)")
        
        return X, y


# =============================================================================
# ULTIMATE SVM TRAINER
# =============================================================================
class UltimateRecognizerTrainer:
    """Train SVM with Grid Search optimization"""
    
    def __init__(self, config: UltimateRecognizerConfig):
        self.config = config
        self.svm = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train SVM model with optimal hyperparameters"""
        print("\n" + "=" * 70)
        print("🎓 TRAINING SVM RECOGNIZER (ULTIMATE VERSION)")
        print("=" * 70)
        
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(self.config.SVM_KERNEL)
        
        print(f"\n⚙️  SVM Configuration:")
        print(f"   Kernel: RBF (Radial Basis Function)")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Number of classes: {len(self.config.CLASS_DIRS)}")
        print(f"   Feature dimension: {X_train.shape[1]}")
        
        start_time = time.time()
        
        if self.config.USE_AUTO_TRAIN:
            print(f"\n🔍 AUTO-TRAINING MODE (Grid Search)")
            print(f"   Cross-Validation: {self.config.KFOLD}-fold")
            print(f"   ⚠️  This will take A LONG TIME")
            print(f"   ⚠️  Please be patient...")
            print(f"   ⏳ Starting Grid Search...")
            
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
        
        # Overall accuracy
        accuracy = np.mean(y_pred == y_test) * 100
        print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
        
        # Classification report
        target_names = [self.config.CLASS_DIRS[i] for i in sorted(self.config.CLASS_DIRS.keys())]
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
        
        # Confusion matrix
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        
        # Print header
        print("\n" + " " * 32 + "Predicted")
        print(" " * 10, end="")
        for i in range(len(target_names)):
            print(f"{i:6d}", end="")
        print()
        
        # Print matrix
        print("Actual")
        for i, class_name in enumerate(target_names):
            print(f"[{i}] {class_name[:20]:20s}", end="")
            for j in range(len(target_names)):
                print(f"{cm[i,j]:6d}", end="")
            print()
        
        # Per-class accuracy
        print("\n📈 Per-Class Accuracy:")
        class_accuracies = []
        for i, class_name in enumerate(target_names):
            if np.sum(cm[i, :]) > 0:
                class_acc = cm[i, i] / np.sum(cm[i, :]) * 100
                class_accuracies.append(class_acc)
                status = "✅" if class_acc >= 85 else "⚠️"
                print(f"   {status} [{i}] {class_name:30s}: {class_acc:.2f}%")
        
        # Overall evaluation
        min_class_acc = min(class_accuracies) if class_accuracies else 0
        print(f"\n📊 Summary:")
        print(f"   Overall Accuracy: {accuracy:.2f}%")
        print(f"   Lowest Class Accuracy: {min_class_acc:.2f}%")
        
        if accuracy >= 90 and min_class_acc >= 85:
            print(f"\n✅✅✅ EXCELLENT! Model meets all target metrics!")
        elif accuracy >= 85:
            print(f"\n✅ GOOD! Model performs well.")
        else:
            print(f"\n⚠️  Model needs improvement. Consider:")
            print(f"      - Collecting more training data for weak classes")
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
        print(f"   Number of classes: {len(self.config.CLASS_DIRS)}")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 70)
    print("🚀 ULTIMATE TRAFFIC SIGN RECOGNIZER TRAINING")
    print("=" * 70)
    print("Features:")
    print("  ✅ Color HOG (B, G, R channels)")
    print("  ✅ Grid Search + Auto-Training")
    print("  ✅ Data Augmentation")
    print("  ✅ Train/Test Split & Evaluation")
    print("=" * 70)
    
    # Initialize configuration
    config = UltimateRecognizerConfig()
    
    # Print configuration
    print("\n📝 Configuration:")
    print(f"   Dataset root: {config.DATASET_ROOT}")
    print(f"   Number of classes: {len(config.CLASS_DIRS)}")
    print(f"   Output model: {config.OUTPUT_MODEL_PATH}")
    print(f"   Test split: {config.TEST_SPLIT * 100:.0f}%")
    print(f"   Augmentation: {'✅ Enabled' if config.USE_AUGMENTATION else '❌ Disabled'}")
    print(f"   Auto-Training: {'✅ Enabled (Grid Search)' if config.USE_AUTO_TRAIN else '❌ Disabled (Manual)'}")
    
    print("\n📋 Classes:")
    for class_id, class_name in config.CLASS_DIRS.items():
        print(f"   [{class_id}] {class_name}")
    
    try:
        # Load dataset
        loader = RecognizerDatasetLoader(config)
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
        print(f"   Test set: {len(X_test)} samples")
        
        # Train model
        trainer = UltimateRecognizerTrainer(config)
        trainer.train(X_train, y_train)
        
        # Evaluate model
        trainer.evaluate(X_test, y_test)
        
        # Save model
        trainer.save_model()
        
        # Print label map for reference
        print("\n" + "=" * 70)
        print("📋 LABEL MAP (Save this for detection pipeline!)")
        print("=" * 70)
        for class_id, class_name in sorted(config.CLASS_DIRS.items()):
            print(f"   {class_id}: \"{class_name}\"")
        print("=" * 70)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📁 Model saved: {config.OUTPUT_MODEL_PATH}")
        print("🎯 Next step: Use both models in your detection pipeline!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()