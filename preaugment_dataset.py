"""
PRE-AUGMENTATION SCRIPT
Purpose: Augment dataset ONCE and save to disk
Benefits:
  - Save RAM during training
  - Faster training (no augmentation overhead)
  - Reusable augmented dataset
  - Can augment large dataset without OOM
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import shutil


# =============================================================================
# CONFIGURATION
# =============================================================================
class PreAugmentConfig:
    """Configuration for pre-augmentation"""
    
    # Augmentation parameters
    AUGMENT_ROTATIONS = [-15, -10, -5, 5, 10, 15]
    AUGMENT_SCALES = [0.9, 0.95, 1.05, 1.1]
    AUGMENT_BRIGHTNESS = [-30, -15, 15, 30]
    AUGMENT_FLIPS = True  # For symmetric signs only
    
    # Processing
    BATCH_SIZE = 100  # Process 100 images at a time to save memory
    IMAGE_FORMAT = "jpg"  # jpg or png
    JPEG_QUALITY = 95  # For jpg format


# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================
class Augmenter:
    """Image augmentation functions"""
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle (degrees)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    @staticmethod
    def scale(image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale image"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(image, (new_w, new_h))
        
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
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
        """Adjust brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255).astype(np.uint8)
        final_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def flip_horizontal(image: np.ndarray) -> np.ndarray:
        """Flip image horizontally"""
        return cv2.flip(image, 1)


# =============================================================================
# PRE-AUGMENTATION ENGINE
# =============================================================================
class PreAugmentEngine:
    """Pre-augment dataset and save to disk"""
    
    def __init__(self, config: PreAugmentConfig):
        self.config = config
        self.augmenter = Augmenter()
        self.stats = {
            'original_count': 0,
            'augmented_count': 0,
            'total_saved': 0
        }
    
    def augment_image(self, image: np.ndarray, allow_flip: bool = True) -> list:
        """Generate all augmented versions of an image"""
        augmented = [('original', image.copy())]
        
        # Rotations
        for angle in self.config.AUGMENT_ROTATIONS:
            aug_img = self.augmenter.rotate(image, angle)
            augmented.append((f'rot{angle}', aug_img))
        
        # Scales
        for scale in self.config.AUGMENT_SCALES:
            aug_img = self.augmenter.scale(image, scale)
            scale_str = str(scale).replace('.', 'p')
            augmented.append((f'scale{scale_str}', aug_img))
        
        # Brightness
        for brightness in self.config.AUGMENT_BRIGHTNESS:
            aug_img = self.augmenter.adjust_brightness(image, brightness)
            augmented.append((f'bright{brightness}', aug_img))
        
        # Flip (optional)
        if allow_flip and self.config.AUGMENT_FLIPS:
            aug_img = self.augmenter.flip_horizontal(image)
            augmented.append(('flip', aug_img))
        
        return augmented
    
    def save_image(self, image: np.ndarray, output_path: str):
        """Save image to disk"""
        if self.config.IMAGE_FORMAT == 'jpg':
            cv2.imwrite(output_path, image, 
                       [cv2.IMWRITE_JPEG_QUALITY, self.config.JPEG_QUALITY])
        else:
            cv2.imwrite(output_path, image)
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         allow_flip: bool = True):
        """Process all images in a directory"""
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(input_dir).glob(ext))
        
        if not image_files:
            print(f"⚠️  No images found in {input_dir}")
            return
        
        total_images = len(image_files)
        self.stats['original_count'] += total_images
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📂 Processing: {input_dir}")
        print(f"   Images: {total_images}")
        print(f"   Output: {output_dir}")
        
        # Process images with progress bar
        saved_count = 0
        for img_path in tqdm(image_files, desc="   Augmenting"):
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            # Get base filename without extension
            base_name = img_path.stem
            
            # Augment
            augmented_images = self.augment_image(img, allow_flip)
            
            # Save all versions
            for aug_name, aug_img in augmented_images:
                filename = f"{base_name}_{aug_name}.{self.config.IMAGE_FORMAT}"
                output_path = os.path.join(output_dir, filename)
                self.save_image(aug_img, output_path)
                saved_count += 1
                self.stats['augmented_count'] += 1
        
        self.stats['total_saved'] += saved_count
        
        multiplier = saved_count / total_images if total_images > 0 else 0
        print(f"   ✅ Saved {saved_count} images (×{multiplier:.1f})")
    
    def save_metadata(self, output_dir: str):
        """Save augmentation metadata"""
        metadata = {
            'config': {
                'rotations': self.config.AUGMENT_ROTATIONS,
                'scales': self.config.AUGMENT_SCALES,
                'brightness': self.config.AUGMENT_BRIGHTNESS,
                'flips': self.config.AUGMENT_FLIPS,
                'format': self.config.IMAGE_FORMAT
            },
            'stats': self.stats
        }
        
        metadata_path = os.path.join(output_dir, 'augmentation_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Metadata saved: {metadata_path}")


# =============================================================================
# DETECTOR DATASET PRE-AUGMENTATION
# =============================================================================
def preaugment_detector_dataset(positive_dir: str, negative_dir: str, 
                                output_dir: str, augment_negatives: bool = False):
    """Pre-augment detector dataset (positive + negative)"""
    
    print("\n" + "=" * 70)
    print("🎨 PRE-AUGMENTING DETECTOR DATASET")
    print("=" * 70)
    
    config = PreAugmentConfig()
    engine = PreAugmentEngine(config)
    
    # Create output directories
    output_positive_dir = os.path.join(output_dir, 'positives')
    output_negative_dir = os.path.join(output_dir, 'negatives')
    
    # Process positive samples (signs - always augment)
    print("\n1️⃣  POSITIVE SAMPLES (Signs)")
    engine.process_directory(positive_dir, output_positive_dir, allow_flip=False)
    
    # Process negative samples (background - optional)
    if augment_negatives:
        print("\n2️⃣  NEGATIVE SAMPLES (Background) - WITH AUGMENTATION")
        # Use minimal augmentation for negatives
        config_neg = PreAugmentConfig()
        config_neg.AUGMENT_ROTATIONS = [-10, 10]
        config_neg.AUGMENT_BRIGHTNESS = [-20, 20]
        config_neg.AUGMENT_SCALES = []
        config_neg.AUGMENT_FLIPS = False
        
        engine_neg = PreAugmentEngine(config_neg)
        engine_neg.process_directory(negative_dir, output_negative_dir, allow_flip=False)
    else:
        print("\n2️⃣  NEGATIVE SAMPLES (Background) - COPYING ONLY")
        print("   (No augmentation - background already diverse)")
        
        # Just copy negative samples without augmentation
        os.makedirs(output_negative_dir, exist_ok=True)
        negative_files = list(Path(negative_dir).glob("*.jpg")) + \
                        list(Path(negative_dir).glob("*.png")) + \
                        list(Path(negative_dir).glob("*.jpeg"))
        
        for img_path in tqdm(negative_files, desc="   Copying"):
            output_path = os.path.join(output_negative_dir, img_path.name)
            shutil.copy2(str(img_path), output_path)
        
        print(f"   ✅ Copied {len(negative_files)} images")
    
    # Save metadata
    engine.save_metadata(output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 AUGMENTATION SUMMARY")
    print("=" * 70)
    print(f"Original images: {engine.stats['original_count']}")
    print(f"Augmented images: {engine.stats['total_saved']}")
    
    if augment_negatives:
        multiplier = engine.stats['total_saved'] / engine.stats['original_count']
    else:
        # Calculate multiplier for positives only
        pos_count = len(list(Path(positive_dir).glob("*.*")))
        aug_pos_count = len(list(Path(output_positive_dir).glob("*.*")))
        neg_count = len(negative_files)
        multiplier = (aug_pos_count + neg_count) / (pos_count + neg_count)
    
    print(f"Overall multiplier: ×{multiplier:.1f}")
    print(f"\n✅ Pre-augmented dataset saved to: {output_dir}")
    print("=" * 70)


# =============================================================================
# RECOGNIZER DATASET PRE-AUGMENTATION
# =============================================================================
def preaugment_recognizer_dataset(input_dir: str, output_dir: str):
    """Pre-augment recognizer dataset (multi-class)"""
    
    print("\n" + "=" * 70)
    print("🎨 PRE-AUGMENTING RECOGNIZER DATASET")
    print("=" * 70)
    
    config = PreAugmentConfig()
    engine = PreAugmentEngine(config)
    
    # Get all class directories
    class_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"❌ No class directories found in {input_dir}")
        return
    
    print(f"\nFound {len(class_dirs)} classes")
    
    # No-flip signs (directional)
    no_flip_classes = ['cam_queo_trai', 'huong_ben_phai']
    
    # Process each class
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        output_class_dir = os.path.join(output_dir, class_name)
        
        # Check if should flip
        allow_flip = class_name not in no_flip_classes
        
        engine.process_directory(str(class_dir), output_class_dir, allow_flip)
    
    # Save metadata
    engine.save_metadata(output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 AUGMENTATION SUMMARY")
    print("=" * 70)
    print(f"Original images: {engine.stats['original_count']}")
    print(f"Augmented images: {engine.stats['total_saved']}")
    multiplier = engine.stats['total_saved'] / engine.stats['original_count']
    print(f"Overall multiplier: ×{multiplier:.1f}")
    print(f"\n✅ Pre-augmented dataset saved to: {output_dir}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Pre-augment dataset and save to disk')
    parser.add_argument('mode', choices=['detector', 'recognizer'],
                       help='Dataset type to augment')
    parser.add_argument('--input-dir', required=True,
                       help='Input directory (for recognizer) or parent dir (for detector)')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for augmented images')
    parser.add_argument('--positive-dir',
                       help='Positive samples directory (for detector mode)')
    parser.add_argument('--negative-dir',
                       help='Negative samples directory (for detector mode)')
    parser.add_argument('--augment-negatives', action='store_true',
                       help='Also augment negative samples (for detector mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'detector':
        if not args.positive_dir or not args.negative_dir:
            print("❌ For detector mode, provide --positive-dir and --negative-dir")
            return
        
        preaugment_detector_dataset(
            args.positive_dir,
            args.negative_dir,
            args.output_dir,
            args.augment_negatives
        )
    
    elif args.mode == 'recognizer':
        preaugment_recognizer_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    # Example usage:
    # python preaugment_dataset.py detector --positive-dir temp_sorted/positives --negative-dir temp_sorted/negatives --output-dir dataset_augmented
    # python preaugment_dataset.py recognizer --input-dir dataset_recognizer_custom --output-dir dataset_recognizer_augmented
    
    main()