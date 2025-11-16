# Debug Video Feature - Usage Guide

## Overview
The debug video feature allows you to save a video output that shows **all detected candidates BEFORE SVM verification**. This is extremely useful for debugging and analyzing the detection pipeline.

## How to Enable

### Method 1: Edit the Configuration in Code
Open `traffic_sign_optimized.py` and locate the `TrafficSignConfig` class (around line 240):

```python
class TrafficSignConfig:
    def __init__(self, auto_detect: bool = True):
        # ... other settings ...
        
        # --- Debug video output (pre-SVM detection) ---
        self.SAVE_DEBUG_VIDEO = True  # Change this to True
        self.DEBUG_VIDEO_PATH = DRIVE_PATH + 'videos/debug_pre_svm_detection.mp4'
```

### Method 2: Modify After Instantiation
In the `main()` function, after creating the config:

```python
config = TrafficSignConfig(auto_detect=True)
config.SAVE_DEBUG_VIDEO = True
config.DEBUG_VIDEO_PATH = 'videos/my_debug_output.mp4'
```

## What You'll See

The debug video includes:

### Visual Elements:
- **Color-coded bounding boxes** around ALL detected candidates:
  - 🔵 Blue boxes = Blue channel detections
  - 🔴 Red boxes = Red channel detections
  - 🟡 Yellow boxes = Yellow channel detections

- **Detailed metrics** for each detection:
  - Color type (BLUE/RED/YELLOW)
  - Area size
  - Circularity (for circular signs)
  - Solidity (for triangular signs)

- **ROI boxes** showing the region of interest for each color channel

- **Frame information**:
  - Current frame number
  - Total pre-SVM detections count

### Debug Information Format:
```
Each detection shows:
BLUE          (Color type)
Area:1250     (Pixel area)
Circ:0.85     (Circularity metric)
```

## Use Cases

### 1. Compare Pre-SVM vs Post-SVM Results
- Enable debug video to see ALL candidates
- Compare with final output video to see what SVM filtered out
- Helps identify if detection is too aggressive or too conservative

### 2. Tune Color Detection Parameters
- See which color ranges are being detected
- Adjust HSV thresholds in `COLOR_PARAMS`
- Verify ROI settings are correct

### 3. Debug False Positives
- Identify why certain objects are being detected
- Check if shape metrics need adjustment
- Analyze problem frames in detail

### 4. Analyze Detection Pipeline
- Understand the full detection flow
- See raw detection before tracking and smoothing
- Evaluate detection quality frame-by-frame

## Performance Impact

⚠️ **Warning:** Enabling debug video will:
- Increase processing time by ~10-20%
- Require additional memory to store full frames
- Generate a large video file (similar size to output)

**Recommendations:**
- Only enable when debugging
- Process a limited number of frames (`MAX_FRAME_ID`)
- Disable for production runs

## Output Location

Default path: `videos/debug_pre_svm_detection.mp4`

You can change this by modifying:
```python
config.DEBUG_VIDEO_PATH = 'your/custom/path/debug.mp4'
```

## Example Workflow

1. **Initial run** - Enable debug video to see all detections
2. **Analyze** - Compare debug video with final output
3. **Tune parameters** - Adjust detection settings
4. **Re-run** - Verify improvements
5. **Disable** - Turn off for final production run

## Technical Details

- Debug video is created during **Pass 1** (Detection + Tracking phase)
- Shows detections BEFORE:
  - SVM Detector verification
  - SVM Recognizer classification
  - Final output rendering
  
- Uses the `draw_debug_detections()` method in the Visualizer class
- Maintains same frame rate and resolution as input video

## Troubleshooting

### Debug video is not created
- Check that `SAVE_DEBUG_VIDEO = True`
- Verify the output directory exists
- Check disk space availability

### Video is empty or corrupted
- Ensure video codec is installed (`opencv-python` with ffmpeg)
- Try changing fourcc codec in the code
- Check console for error messages

### Processing is too slow
- Reduce `MAX_FRAME_ID` to process fewer frames
- Decrease batch size temporarily
- Disable other outputs (mask videos)

## Related Files

- `traffic_sign_optimized.py` - Main processing script
- `videos/video_output1.mp4` - Final output (post-SVM)
- `videos/debug_pre_svm_detection.mp4` - Debug output (pre-SVM)
- `videos/mask_video_*.mp4` - Color mask videos (if enabled)
