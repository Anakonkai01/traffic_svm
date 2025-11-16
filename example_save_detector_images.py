"""
EXAMPLE: How to Save SVM Detector Verified Images

This example shows how to enable saving all images that the SVM detector
classifies as traffic signs to the "check_data" folder.

This is useful for:
- Analyzing false positives from the detector
- Building a dataset of detected signs
- Manual verification of detector performance
- Debugging detector accuracy issues
"""

import sys
sys.path.append('traffic_main_pipeline')
from traffic_sign_optimized import TrafficSignConfig, run_traffic_sign_detection

def main():
    # Create configuration
    config = TrafficSignConfig(auto_detect=True)
    
    # Configure input/output paths
    config.INPUT_VIDEO_PATH = 'videos/video1.mp4'
    config.OUTPUT_VIDEO_PATH = 'videos/video_output1.mp4'
    
    # ===================================================================
    # ENABLE SAVING DETECTOR VERIFIED IMAGES
    # ===================================================================
    config.SAVE_DETECTOR_IMAGES = True
    config.DETECTOR_IMAGES_FOLDER = 'check_data'  # Folder will be created automatically
    
    # Optional: Also enable debug video to see pre-SVM detections
    config.SAVE_DEBUG_VIDEO = True
    config.DEBUG_VIDEO_PATH = 'videos/debug_pre_svm_detection.mp4'
    
    print("\n" + "="*70)
    print("CONFIGURATION:")
    print("="*70)
    print(f"✓ Input video: {config.INPUT_VIDEO_PATH}")
    print(f"✓ Output video: {config.OUTPUT_VIDEO_PATH}")
    print(f"✓ Save detector images: {config.SAVE_DETECTOR_IMAGES}")
    print(f"✓ Detector images folder: {config.DETECTOR_IMAGES_FOLDER}")
    print(f"✓ Save debug video: {config.SAVE_DEBUG_VIDEO}")
    print("="*70)
    
    # Run detection
    run_traffic_sign_detection()
    
    print("\n" + "="*70)
    print("CHECK YOUR RESULTS:")
    print("="*70)
    print(f"📁 All images classified as signs by SVM detector:")
    print(f"   → {config.DETECTOR_IMAGES_FOLDER}/")
    print(f"\n   Images are named: frame######_color_xXXX_yYYY.jpg")
    print(f"   - frame######: Frame number in video")
    print(f"   - color: Detection color (blue/red/yellow)")
    print(f"   - xXXX_yYYY: Position in frame")
    print(f"\n💡 Review these images to identify false positives!")
    print("="*70)

if __name__ == '__main__':
    main()
