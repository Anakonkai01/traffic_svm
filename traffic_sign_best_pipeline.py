import cv2
import numpy as np
import time
import gc
import psutil
import platform
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool, Manager, cpu_count
from concurrent.futures import ThreadPoolExecutor
import queue
from functools import partial


DRIVE_PATH = ""

MODEL_DETECTOR_PATH = DRIVE_PATH + "svm_sign_detector_hog_color_v1.xml"
MODEL_RECOGNIZER_PATH = DRIVE_PATH + "svm_sign_recognizer_hog_color_v1.xml"


# =============================================================================
# CONFIGURATION CLASS - HIGH ACCURACY PIPELINE
# =============================================================================
class HighAccuracyConfig:
    """Configuration for high-accuracy detection pipeline"""
    
    def __init__(self):
        # Video Paths
        self.INPUT_VIDEO_PATH = "videos/video2.mp4"
        self.OUTPUT_VIDEO_PATH = "videos/video_2_output_high_accuracy.mp4"
        
        # Image Pyramid Settings
        self.PYRAMID_SCALES = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3]  # Multi-scale detection
        self.MIN_SIGN_SIZE = 32  # Minimum sign size in pixels
        self.MAX_SIGN_SIZE = 200  # Maximum sign size in pixels
        
        # Sliding Window Settings
        self.WINDOW_SIZE = (64, 64)  # Fixed window size for HOG
        self.STEP_SIZE = 8  # Pixels to move window (smaller = more thorough, slower)
        
        # Non-Maximum Suppression Settings
        self.NMS_OVERLAP_THRESHOLD = 0.3  # IoU threshold for NMS
        
        # HOG Feature Settings
        self.HOG_WIN_SIZE = (64, 64)
        self.HOG_BLOCK_SIZE = (16, 16)
        self.HOG_BLOCK_STRIDE = (8, 8)
        self.HOG_CELL_SIZE = (8, 8)
        self.HOG_NBINS = 9
        
        # Color HOG Settings (for detector)
        self.USE_COLOR_HOG = True
        
        # Processing Settings
        self.NUM_WORKERS = max(1, cpu_count() - 1)
        self.BATCH_SIZE = 4  # Process fewer frames at once (high accuracy mode)
        self.PREFETCH_FRAMES = 16
        
        # ROI Crop Settings
        self.TOP_CROP_PERCENT = 0.35
        self.RIGHT_CROP_PERCENT = 0.05
        
        # Progress Settings
        self.PROGRESS_UPDATE_INTERVAL = 30
        self.MAX_FRAME_ID = 999999
        
        # GPU Settings
        self.USE_GPU = True  # Enable GPU acceleration if available
        
        # Visualization
        self.SHOW_ALL_PYRAMID_DETECTIONS = False  # Debug: show detections at all scales


# =============================================================================
# AUTO-DETECTION SYSTEM
# =============================================================================
class HardwareDetector:
    """Automatically detects and analyzes computer hardware"""
    
    def __init__(self):
        self.cpu_cores = cpu_count()
        self.ram_gb = self._get_ram_gb()
        self.has_gpu = self._check_gpu()
        self.storage_type = self._detect_storage_type()
        self.cpu_model = self._get_cpu_model()
        self.os_info = self._get_os_info()
        
    def _get_ram_gb(self) -> float:
        """Get total RAM in GB"""
        try:
            return psutil.virtual_memory().total / (1024 ** 3)
        except:
            return 8.0
    
    def _check_gpu(self) -> bool:
        """Check if CUDA-capable GPU is available"""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False
    
    def _detect_storage_type(self) -> str:
        """Detect if storage is SSD or HDD"""
        try:
            if platform.system() == 'Windows':
                import subprocess
                output = subprocess.check_output(
                    'powershell "Get-PhysicalDisk | Select MediaType"',
                    shell=True,
                    text=True
                )
                if 'SSD' in output:
                    return 'SSD'
                elif 'NVMe' in output:
                    return 'NVMe'
                else:
                    return 'HDD'
            elif platform.system() == 'Linux':
                try:
                    with open('/sys/block/sda/queue/rotational', 'r') as f:
                        if f.read().strip() == '0':
                            return 'SSD'
                        else:
                            return 'HDD'
                except:
                    return 'SSD'
            else:
                return 'SSD'
        except:
            return 'SSD'
    
    def _get_cpu_model(self) -> str:
        """Get CPU model name"""
        try:
            if platform.system() == 'Windows':
                return platform.processor()
            elif platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
            else:
                return platform.processor()
        except:
            return "Unknown CPU"
    
    def _get_os_info(self) -> str:
        """Get OS information"""
        return f"{platform.system()} {platform.release()}"
    
    def get_optimal_config_high_accuracy(self) -> Dict:
        """Calculate optimal configuration for high-accuracy pipeline"""
        config = {}
        
        # For high accuracy, we prioritize thoroughness over speed
        # Reduce parallelism slightly to avoid memory issues
        if self.cpu_cores <= 2:
            config['num_workers'] = 1
            config['batch_size'] = 2
            config['pyramid_scales'] = [1.0, 0.7, 0.5]
            config['step_size'] = 12
        elif self.cpu_cores <= 4:
            config['num_workers'] = 2
            config['batch_size'] = 2
            config['pyramid_scales'] = [1.0, 0.8, 0.65, 0.5]
            config['step_size'] = 10
        elif self.cpu_cores <= 8:
            config['num_workers'] = max(2, self.cpu_cores - 2)
            config['batch_size'] = 4
            config['pyramid_scales'] = [1.0, 0.8, 0.65, 0.5, 0.4]
            config['step_size'] = 8
        else:
            config['num_workers'] = max(2, self.cpu_cores - 2)
            config['batch_size'] = 4
            config['pyramid_scales'] = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3]
            config['step_size'] = 8
        
        # RAM considerations
        if self.ram_gb < 6:
            config['batch_size'] = 1
            config['pyramid_scales'] = config['pyramid_scales'][:3]
        elif self.ram_gb < 12:
            config['batch_size'] = max(1, config['batch_size'] // 2)
        
        config['use_gpu'] = self.has_gpu
        
        return config
    
    def print_hardware_report(self):
        """Print detailed hardware report"""
        print("\n" + "=" * 70)
        print("🔍 HARDWARE DETECTION REPORT (HIGH ACCURACY MODE)")
        print("=" * 70)
        print(f"💻 System: {self.os_info}")
        print(f"🧠 CPU: {self.cpu_model}")
        print(f"⚙️  Cores: {self.cpu_cores} cores")
        print(f"🎮 RAM: {self.ram_gb:.1f} GB")
        print(f"💾 Storage: {self.storage_type}")
        print(f"🎨 GPU (CUDA): {'✅ Available' if self.has_gpu else '❌ Not detected'}")
        print("=" * 70)
    
    def print_optimized_settings(self, config: Dict):
        """Print the auto-configured settings"""
        print("\n⚙️  HIGH ACCURACY SETTINGS:")
        print("=" * 70)
        print(f"   Worker Processes: {config['num_workers']}")
        print(f"   Batch Size: {config['batch_size']} frames")
        print(f"   Pyramid Scales: {len(config['pyramid_scales'])} levels")
        print(f"   Window Step Size: {config['step_size']} pixels")
        print(f"   GPU Acceleration: {'✅ Enabled' if config['use_gpu'] else '❌ Disabled'}")
        print("=" * 70)


# =============================================================================
# COLOR HOG EXTRACTOR
# =============================================================================
class ColorHOGExtractor:
    """Extract HOG features from color channels"""
    
    def __init__(self, win_size=(64, 64), block_size=(16, 16), 
                 block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        self.win_size = win_size
        self.hog = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size, nbins
        )
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract Color HOG features"""
        if image.shape[:2] != self.win_size:
            image = cv2.resize(image, self.win_size)
        
        # Split into B, G, R channels
        b, g, r = cv2.split(image)
        
        # Extract HOG from each channel
        hog_b = self.hog.compute(b).flatten()
        hog_g = self.hog.compute(g).flatten()
        hog_r = self.hog.compute(r).flatten()
        
        # Concatenate all channels
        color_hog = np.concatenate([hog_b, hog_g, hog_r])
        
        return color_hog.astype(np.float32)


# =============================================================================
# SVM DETECTOR (Model B)
# =============================================================================
class SVMDetector:
    """SVM-based sign detector using Color HOG"""
    
    def __init__(self, model_path: str, config: HighAccuracyConfig):
        self.model_path = model_path
        self.config = config
        self.svm = None
        self.hog_extractor = ColorHOGExtractor(
            win_size=config.HOG_WIN_SIZE,
            block_size=config.HOG_BLOCK_SIZE,
            block_stride=config.HOG_BLOCK_STRIDE,
            cell_size=config.HOG_CELL_SIZE,
            nbins=config.HOG_NBINS
        )
        self.load_model()
    
    def load_model(self):
        """Load SVM detector model"""
        try:
            self.svm = cv2.ml.SVM_load(self.model_path)
            print(f"✅ Loaded SVM Detector: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading SVM Detector: {e}")
            raise
    
    def detect_sliding_window(self, image: np.ndarray, scale: float = 1.0) -> List[Tuple]:
        """
        Detect signs using sliding window on a single scale
        Returns: List of (bbox, confidence) tuples
        """
        detections = []
        h, w = image.shape[:2]
        win_w, win_h = self.config.WINDOW_SIZE
        step = self.config.STEP_SIZE
        
        # Slide window across image
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                # Extract window
                window = image[y:y+win_h, x:x+win_w]
                
                # Extract HOG features
                features = self.hog_extractor.extract(window)
                features = features.reshape(1, -1)
                
                # Predict with SVM
                _, result = self.svm.predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                confidence = -result[0][0]  # Distance to hyperplane
                
                # Threshold
                if confidence > 0:  # Positive class
                    # Convert bbox back to original scale
                    bbox = (
                        int(x / scale),
                        int(y / scale),
                        int(win_w / scale),
                        int(win_h / scale)
                    )
                    detections.append((bbox, confidence))
        
        return detections
    
    def verify(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Verify a single detection (used in Pass 2)"""
        x, y, w, h = bbox
        
        # Crop region
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            return False
        
        region = image[y:y+h, x:x+w]
        
        # Extract features
        features = self.hog_extractor.extract(region)
        features = features.reshape(1, -1)
        
        # Predict
        _, result = self.svm.predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        confidence = -result[0][0]
        
        return confidence > 0


# =============================================================================
# SVM RECOGNIZER (Model A)
# =============================================================================
class SVMRecognizer:
    """SVM-based sign recognizer using Color HOG"""
    
    def __init__(self, model_path: str, config: HighAccuracyConfig):
        self.model_path = model_path
        self.config = config
        self.svm = None
        self.hog_extractor = ColorHOGExtractor(
            win_size=config.HOG_WIN_SIZE,
            block_size=config.HOG_BLOCK_SIZE,
            block_stride=config.HOG_BLOCK_STRIDE,
            cell_size=config.HOG_CELL_SIZE,
            nbins=config.HOG_NBINS
        )
        self.load_model()
        
        # Sign name mapping
        self.sign_names = {
            0: "cam_di_nguoc_chieu",
            1: "cam_queo_trai",
            2: "cam_do_xe",
            3: "cam_dung_do_xe",
            4: "huong_ben_phai",
            5: "canh_bao_di_cham",
            6: "canh_bao_nguoi_qua_duong",
            7: "canh_bao_duong_gap_khuc"
        }
    
    def load_model(self):
        """Load SVM recognizer model"""
        try:
            self.svm = cv2.ml.SVM_load(self.model_path)
            print(f"✅ Loaded SVM Recognizer: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading SVM Recognizer: {e}")
            raise
    
    def recognize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple]:
        """Recognize a single sign"""
        x, y, w, h = bbox
        
        # Crop region
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            return None
        
        region = image[y:y+h, x:x+w]
        
        # Extract features
        features = self.hog_extractor.extract(region)
        features = features.reshape(1, -1)
        
        # Predict
        label = int(self.svm.predict(features)[1][0][0])
        sign_name = self.sign_names.get(label, "unknown")
        
        return (bbox, sign_name, label)
    
    def recognize_batch(self, image: np.ndarray, detections: List[Tuple]) -> List[Tuple]:
        """Recognize multiple signs"""
        results = []
        for detection in detections:
            bbox = detection[0] if isinstance(detection, tuple) and len(detection) > 1 else detection
            result = self.recognize(image, bbox)
            if result:
                results.append(result)
        return results


# =============================================================================
# NON-MAXIMUM SUPPRESSION
# =============================================================================
def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute IoU (Intersection over Union) between two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(detections: List[Tuple], overlap_threshold: float = 0.3) -> List[Tuple]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections
    Input: List of (bbox, confidence)
    Output: List of (bbox, confidence) after NMS
    """
    if not detections:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    
    keep = []
    
    while detections:
        # Take detection with highest confidence
        best = detections[0]
        keep.append(best)
        detections = detections[1:]
        
        # Remove overlapping detections
        filtered = []
        for det in detections:
            iou = compute_iou(best[0], det[0])
            if iou < overlap_threshold:
                filtered.append(det)
        
        detections = filtered
    
    return keep


# =============================================================================
# IMAGE PYRAMID GENERATOR
# =============================================================================
def create_image_pyramid(image: np.ndarray, scales: List[float]) -> List[Tuple[np.ndarray, float]]:
    """
    Create image pyramid with multiple scales
    Returns: List of (scaled_image, scale) tuples
    """
    pyramid = []
    
    for scale in scales:
        if scale == 1.0:
            pyramid.append((image.copy(), scale))
        else:
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h > 0 and new_w > 0:
                scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                pyramid.append((scaled, scale))
    
    return pyramid


# =============================================================================
# PARALLEL FRAME READER
# =============================================================================
class ParallelFrameReader:
    """Multi-threaded frame reader with prefetching"""
    
    def __init__(self, video_path: str, total_frames: int, prefetch_size: int = 32):
        self.video_path = video_path
        self.total_frames = total_frames
        self.prefetch_size = prefetch_size
        self.frame_queue = queue.Queue(maxsize=prefetch_size)
        self.stop_flag = False
        self.thread = None
    
    def _reader_thread(self):
        """Background thread that reads frames"""
        cap = cv2.VideoCapture(self.video_path)
        frame_num = 0
        
        while not self.stop_flag and frame_num < self.total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_queue.put((frame_num, frame))
            frame_num += 1
        
        cap.release()
        self.frame_queue.put(None)  # Signal end
    
    def start(self):
        """Start reading frames in background"""
        self.stop_flag = False
        self.thread = ThreadPoolExecutor(max_workers=1)
        self.thread.submit(self._reader_thread)
    
    def get_frame(self):
        """Get next frame from queue"""
        return self.frame_queue.get()
    
    def stop(self):
        """Stop reading frames"""
        self.stop_flag = True
        if self.thread:
            self.thread.shutdown(wait=True)


# =============================================================================
# VISUALIZER
# =============================================================================
class Visualizer:
    """Draw detection results on frames"""
    
    def __init__(self, config: HighAccuracyConfig):
        self.config = config
        
        # Colors for different sign types
        self.colors = {
            "cam_di_nguoc_chieu": (0, 0, 255),      # Red
            "cam_queo_trai": (0, 0, 255),           # Red
            "cam_do_xe": (0, 0, 255),               # Red
            "cam_dung_do_xe": (0, 0, 255),          # Red
            "huong_ben_phai": (255, 0, 0),          # Blue
            "canh_bao_di_cham": (0, 255, 255),      # Yellow
            "canh_bao_nguoi_qua_duong": (0, 255, 255), # Yellow
            "canh_bao_duong_gap_khuc": (0, 255, 255),  # Yellow
        }
    
    def draw_all(self, frame: np.ndarray, frame_num: int, 
                 detections: List[Tuple], roi_map=None) -> np.ndarray:
        """Draw all detections on frame"""
        output = frame.copy()
        
        for detection in detections:
            bbox, sign_name, label = detection
            x, y, w, h = bbox
            
            # Get color
            color = self.colors.get(sign_name, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_text = sign_name.replace("_", " ").title()
            cv2.putText(output, label_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw frame number
        cv2.putText(output, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output


# =============================================================================
# WORKER FUNCTION FOR MULTIPROCESSING
# =============================================================================
def process_frame_pyramid_worker(args):
    """
    Worker function to process a single frame with image pyramid and sliding window
    
    Returns: (frame_num, all_detections_after_nms)
    """
    frame_num, frame, config = args
    
    # Initialize detector (each worker needs its own)
    detector = SVMDetector(MODEL_DETECTOR_PATH, config)
    
    # Create image pyramid
    pyramid = create_image_pyramid(frame, config.PYRAMID_SCALES)
    
    # Collect all detections across all scales
    all_detections = []
    
    for scaled_image, scale in pyramid:
        # Detect on this scale
        scale_detections = detector.detect_sliding_window(scaled_image, scale)
        all_detections.extend(scale_detections)
    
    # Apply Non-Maximum Suppression
    final_detections = non_max_suppression(all_detections, config.NMS_OVERLAP_THRESHOLD)
    
    # Return only bboxes (without confidence scores for now)
    bboxes = [det[0] for det in final_detections]
    
    return (frame_num, bboxes)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def main():
    """Main processing pipeline"""
    
    print("\n" + "=" * 70)
    print("🎯 TRAFFIC SIGN DETECTION - HIGH ACCURACY PIPELINE")
    print("=" * 70)
    
    # Initialize configuration
    config = HighAccuracyConfig()
    
    # Hardware detection
    hw_detector = HardwareDetector()
    hw_detector.print_hardware_report()
    
    # Get optimized settings
    hw_config = hw_detector.get_optimal_config_high_accuracy()
    
    # Apply hardware-optimized settings
    config.NUM_WORKERS = hw_config['num_workers']
    config.BATCH_SIZE = hw_config['batch_size']
    config.PYRAMID_SCALES = hw_config['pyramid_scales']
    config.STEP_SIZE = hw_config['step_size']
    config.USE_GPU = hw_config['use_gpu']
    
    hw_detector.print_optimized_settings(hw_config)
    
    print("\n⚠️  HIGH ACCURACY MODE:")
    print("   - Image Pyramid: ✅ Enabled")
    print("   - Sliding Window: ✅ Enabled")
    print("   - Non-Max Suppression: ✅ Enabled")
    print("   - Expected Speed: 🐌 SLOW (priority: accuracy)")
    
    try:
        start_total = time.time()
        
        # Load models
        detector_svm = SVMDetector(MODEL_DETECTOR_PATH, config)
        recognizer_svm = SVMRecognizer(MODEL_RECOGNIZER_PATH, config)
        
        # Initialize visualizer
        visualizer = Visualizer(config)
        
        # Open video
        cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video '{config.INPUT_VIDEO_PATH}'")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate crop dimensions
        h_crop = int(h_orig * (1 - config.TOP_CROP_PERCENT))
        w_crop = int(w_orig * (1 - config.RIGHT_CROP_PERCENT))
        
        print(f"\n📹 Video Info:")
        print(f"   Resolution: {w_orig}x{h_orig}")
        print(f"   Processing Region: {w_crop}x{h_crop}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        
        # =====================================================================
        # PASS 1: PYRAMID + SLIDING WINDOW + NMS DETECTION
        # =====================================================================
        print("\n" + "=" * 70)
        print("🔍 PASS 1: PYRAMID + SLIDING WINDOW + NMS DETECTION")
        print("=" * 70)
        start_detection = time.time()
        
        # Start parallel frame reader
        frame_reader = ParallelFrameReader(config.INPUT_VIDEO_PATH, 
                                          total_frames, 
                                          config.PREFETCH_FRAMES)
        frame_reader.start()
        
        # Storage for all detections
        all_frame_detections = {}
        
        frame_batch = []
        frame_count = 0
        
        # Use multiprocessing pool
        with Pool(processes=config.NUM_WORKERS) as pool:
            while True:
                item = frame_reader.get_frame()
                if item is None:
                    break
                
                frame_num, frame_full = item
                frame_to_process = frame_full[0:h_crop, 0:w_crop]
                
                frame_batch.append((frame_num, frame_to_process, config))
                
                # Process batch when full
                if len(frame_batch) >= config.BATCH_SIZE:
                    results = pool.map(process_frame_pyramid_worker, frame_batch)
                    
                    for frame_num, bboxes in results:
                        all_frame_detections[frame_num] = bboxes
                    
                    frame_count += len(frame_batch)
                    frame_batch.clear()
                    
                    if frame_count % config.PROGRESS_UPDATE_INTERVAL == 0:
                        progress = int(frame_count / total_frames * 100)
                        elapsed = time.time() - start_detection
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"   Detected {frame_count}/{total_frames} frames ({progress}%) - {current_fps:.1f} FPS")
                        gc.collect()
            
            # Process remaining frames
            if frame_batch:
                results = pool.map(process_frame_pyramid_worker, frame_batch)
                for frame_num, bboxes in results:
                    all_frame_detections[frame_num] = bboxes
                
                frame_count += len(frame_batch)
        
        frame_reader.stop()
        
        detection_time = time.time() - start_detection
        print(f"\n✅ Detection Pass completed in {detection_time:.2f}s")
        print(f"   Average Speed: {frame_count/detection_time:.1f} FPS")
        
        # Count total detections
        total_detections = sum(len(dets) for dets in all_frame_detections.values())
        print(f"   Total detections (after NMS): {total_detections}")
        
        # =====================================================================
        # PASS 2: RECOGNIZE + RENDER
        # =====================================================================
        print("\n" + "=" * 70)
        print("🎨 PASS 2: RECOGNIZE + RENDER")
        print("=" * 70)
        start_render = time.time()
        
        # Start parallel frame reader again
        frame_reader = ParallelFrameReader(config.INPUT_VIDEO_PATH, 
                                          total_frames, 
                                          config.PREFETCH_FRAMES)
        frame_reader.start()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps,
                                       (w_orig, h_orig))
        
        if not video_writer.isOpened():
            print(f"❌ Error: Cannot create output '{config.OUTPUT_VIDEO_PATH}'")
            return
        
        frame_count_render = 0
        frames_with_detections = 0
        total_signs_recognized = 0
        
        while True:
            item = frame_reader.get_frame()
            if item is None:
                break
            
            frame_num, frame = item
            
            # Get detections for this frame
            bboxes = all_frame_detections.get(frame_num, [])
            
            # Recognize signs
            recognized_detections = []
            if bboxes:
                recognized_detections = recognizer_svm.recognize_batch(frame, bboxes)
                total_signs_recognized += len(recognized_detections)
            
            # Draw on frame
            frame_output = visualizer.draw_all(frame, frame_num, recognized_detections)
            video_writer.write(frame_output)
            
            if recognized_detections:
                frames_with_detections += 1
            
            frame_count_render += 1
            
            if frame_count_render % config.PROGRESS_UPDATE_INTERVAL == 0:
                progress = int(frame_count_render / total_frames * 100)
                elapsed = time.time() - start_render
                current_fps = frame_count_render / elapsed if elapsed > 0 else 0
                print(f"   Rendered {frame_count_render}/{total_frames} frames ({progress}%) "
                      f"- {current_fps:.1f} FPS - {total_signs_recognized} signs")
            
            del frame, frame_output
        
        frame_reader.stop()
        video_writer.release()
        
        render_time = time.time() - start_render
        print(f"\n✅ Rendering Pass completed in {render_time:.2f}s")
        print(f"   Average Speed: {frame_count_render/render_time:.1f} FPS")
        print(f"   Frames with signs: {frames_with_detections}/{frame_count_render}")
        print(f"   Total signs recognized: {total_signs_recognized}")
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        total_time = time.time() - start_total
        
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"📁 Output: {config.OUTPUT_VIDEO_PATH}")
        
        print(f"\n⏱️  PERFORMANCE SUMMARY:")
        print("=" * 70)
        print(f"   Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"   Detection Phase: {detection_time:.2f}s ({detection_time/total_time*100:.1f}%)")
        print(f"   Rendering Phase: {render_time:.2f}s ({render_time/total_time*100:.1f}%)")
        print(f"   Overall Speed: {frame_count_render/total_time:.1f} FPS")
        
        print(f"\n🎯 HIGH ACCURACY PIPELINE:")
        print(f"   ✅ Image Pyramid ({len(config.PYRAMID_SCALES)} scales)")
        print(f"   ✅ Sliding Window (step: {config.STEP_SIZE}px)")
        print(f"   ✅ Non-Maximum Suppression (IoU: {config.NMS_OVERLAP_THRESHOLD})")
        print(f"   ✅ Color HOG Features")
        print(f"   ✅ Multi-processing ({config.NUM_WORKERS} workers)")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()


if __name__ == "__main__":
    main()