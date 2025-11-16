#!/usr/bin/env python3
"""
QUICK EXAMPLE: How to enable debug video output

This example shows how to run the traffic sign detection
with debug video enabled to see all detections BEFORE SVM filtering.
"""

# This is a reference example - the actual changes should be made in traffic_sign_optimized.py

# ==============================================================================
# OPTION 1: Edit the TrafficSignConfig class directly
# ==============================================================================
# In traffic_sign_optimized.py, find this section (around line 240):

"""
class TrafficSignConfig:
    def __init__(self, auto_detect: bool = True):
        # --- File paths ---
        self.INPUT_VIDEO_PATH = DRIVE_PATH + 'videos/video1.mp4'
        self.OUTPUT_VIDEO_PATH = DRIVE_PATH + 'videos/video_output1.mp4'
        
        # ... other settings ...
        
        # --- Debug video output (pre-SVM detection) ---
        self.SAVE_DEBUG_VIDEO = False  # <-- CHANGE THIS TO True
        self.DEBUG_VIDEO_PATH = DRIVE_PATH + 'videos/debug_pre_svm_detection.mp4'
"""

# ==============================================================================
# OPTION 2: Modify in main() function after creating config
# ==============================================================================
# In the main() function, add these lines after creating config:

"""
def main():
    print("=" * 70)
    print("  TRAFFIC SIGN DETECTION WITH AUTO-OPTIMIZATION")
    print("=" * 70)

    try:
        start_total = time.time()
        
        # AUTO-DETECT HARDWARE AND CONFIGURE
        config = TrafficSignConfig(auto_detect=True)
        
        # ENABLE DEBUG VIDEO
        config.SAVE_DEBUG_VIDEO = True  # <-- ADD THIS LINE
        config.DEBUG_VIDEO_PATH = 'videos/my_debug_output.mp4'  # <-- OPTIONAL: custom path
        
        visualizer = Visualizer(config)
        # ... rest of the code ...
"""

# ==============================================================================
# WHAT TO EXPECT
# ==============================================================================

print("""
When you run the script with debug video enabled, you'll see:

During Pass 1 (Detection):
===========================
⚡ PASS 1: PARALLEL DETECTION + TRACKING
======================================================================

📹 Debug video enabled: videos/debug_pre_svm_detection.mp4

   Detected 100 frames (10%) - 45.2 FPS
   Detected 200 frames (20%) - 46.1 FPS
   ...

✅ Debug video saved: videos/debug_pre_svm_detection.mp4
✅ Detection Pass completed in 22.15s


At the end:
===========
✅ PROCESSING COMPLETE!
======================================================================
📁 Output: videos/video_output1.mp4
📁 Debug (Pre-SVM): videos/debug_pre_svm_detection.mp4

⏱️  PERFORMANCE SUMMARY:
======================================================================
   Total Time: 45.30s (0.76 min)
   Detection Phase: 22.15s (48.9%)
   Rendering Phase: 23.15s (51.1%)
   Overall Speed: 44.1 FPS


The debug video will contain:
==============================
- ALL detected candidates (before SVM verification)
- Color-coded bounding boxes:
  * Blue boxes = Blue channel detections
  * Red boxes = Red channel detections  
  * Yellow boxes = Yellow channel detections
  
- Detailed metrics for each detection:
  * Color type (BLUE/RED/YELLOW)
  * Area (pixel count)
  * Circularity (for circles) or Solidity (for triangles)
  
- ROI visualization (region of interest boxes)
- Frame numbers
- Detection counts

COMPARISON:
===========
- debug_pre_svm_detection.mp4 = ALL candidates detected by color/shape
- video_output1.mp4 = ONLY verified & recognized signs after SVM filtering

The difference shows you what the SVM detector filtered out!
""")

# ==============================================================================
# EXAMPLE: Process a short video for quick debugging
# ==============================================================================

print("""
TIP: For quick debugging, process only the first few frames:

In TrafficSignConfig:
    self.MAX_FRAME_ID = 300  # Process only first 300 frames
    self.SAVE_DEBUG_VIDEO = True
    
This will:
- Process quickly (10-15 seconds)
- Generate smaller debug video
- Allow rapid iteration during tuning
""")
