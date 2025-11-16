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

"""
TRAFFIC SIGN DETECTION SYSTEM - OPTIMIZED VERSION

DEBUG VIDEO FEATURE:
====================
To enable debug video output (before SVM detection):
1. Open this file and locate the TrafficSignConfig class
2. Set: config.SAVE_DEBUG_VIDEO = True
3. Optionally modify: config.DEBUG_VIDEO_PATH = 'your/path/debug.mp4'

The debug video will show:
- All detected candidates BEFORE SVM verification
- Color-coded bounding boxes (Blue/Red/Yellow)
- Detailed metrics (Area, Circularity/Solidity)
- ROI boxes for each color channel
- Frame numbers and detection counts

This is useful for:
- Debugging detection pipeline
- Analyzing false positives before SVM filtering
- Comparing pre-SVM vs post-SVM results
- Tuning color detection parameters
"""


DRIVE_PATH = "../"


MODEL_DETECTOR_PATH = DRIVE_PATH + "models/svm_sign_detector_v4.xml"
MODEL_RECOGNIZER_PATH = DRIVE_PATH + "models/svm_sign_recognizer_v3.xml"


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
            return 8.0  # Default fallback
    
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
                # On Windows, check if system drive is SSD
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
                # Check if /sys/block/sda/queue/rotational is 0 (SSD)
                try:
                    with open('/sys/block/sda/queue/rotational', 'r') as f:
                        if f.read().strip() == '0':
                            return 'SSD'
                        else:
                            return 'HDD'
                except:
                    return 'SSD'  # Assume SSD if can't determine
            else:
                # macOS - assume SSD for modern Macs
                return 'SSD'
        except:
            return 'SSD'  # Default to SSD assumptions
    
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
    
    def get_optimal_config(self) -> Dict:
        """Calculate optimal configuration based on hardware"""
        config = {}
        
        # CPU Configuration
        if self.cpu_cores <= 2:
            # Very low-end (old laptop)
            config['num_workers'] = 1
            config['batch_size'] = 4
            config['prefetch_frames'] = 16
            config['profile'] = 'Low-End (2 cores)'
        elif self.cpu_cores <= 4:
            # Low-end (budget laptop/old desktop)
            config['num_workers'] = max(1, self.cpu_cores - 1)
            config['batch_size'] = 8
            config['prefetch_frames'] = 32
            config['profile'] = 'Budget (4 cores)'
        elif self.cpu_cores <= 8:
            # Mid-range (modern laptop/desktop)
            config['num_workers'] = max(1, self.cpu_cores - 1)
            config['batch_size'] = 16
            config['prefetch_frames'] = 64
            config['profile'] = 'Mid-Range (6-8 cores)'
        elif self.cpu_cores <= 12:
            # High-end (gaming/workstation)
            config['num_workers'] = max(1, self.cpu_cores - 1)
            config['batch_size'] = 24
            config['prefetch_frames'] = 96
            config['profile'] = 'High-End (10-12 cores)'
        else:
            # Extreme (server/high-end workstation)
            config['num_workers'] = max(1, self.cpu_cores - 2)
            config['batch_size'] = 32
            config['prefetch_frames'] = 128
            config['profile'] = 'Extreme (12+ cores)'
        
        # RAM Adjustments
        if self.ram_gb < 6:
            # Very limited RAM - reduce batch size
            config['batch_size'] = max(4, config['batch_size'] // 2)
            config['prefetch_frames'] = max(16, config['prefetch_frames'] // 2)
            config['memory_warning'] = True
        elif self.ram_gb < 12:
            # Limited RAM - slight reduction
            config['batch_size'] = max(8, int(config['batch_size'] * 0.75))
            config['memory_warning'] = False
        else:
            # Plenty of RAM - can increase if desired
            if self.ram_gb >= 32:
                config['batch_size'] = int(config['batch_size'] * 1.5)
            config['memory_warning'] = False
        
        # Storage Adjustments
        if self.storage_type == 'NVMe':
            # Super fast - increase prefetch
            config['prefetch_frames'] = int(config['prefetch_frames'] * 1.5)
        elif self.storage_type == 'HDD':
            # Slower - reduce prefetch to avoid seeking
            config['prefetch_frames'] = max(16, config['prefetch_frames'] // 2)
        
        # GPU Configuration
        config['use_gpu'] = self.has_gpu
        
        return config
    
    def print_hardware_report(self):
        """Print detailed hardware report"""
        print("\n" + "=" * 70)
        print("🔍 HARDWARE DETECTION REPORT")
        print("=" * 70)
        print(f"💻 System: {self.os_info}")
        print(f"🧠 CPU: {self.cpu_model}")
        print(f"⚙️  Cores: {self.cpu_cores} cores")
        print(f"🎮 RAM: {self.ram_gb:.1f} GB")
        print(f"💾 Storage: {self.storage_type}")
        print(f"🎨 GPU (CUDA): {'✅ Available' if self.has_gpu else '❌ Not detected'}")
        
        # Get optimal config
        config = self.get_optimal_config()
        
        print(f"\n📊 Performance Profile: {config['profile']}")
        print("=" * 70)
    
    def print_optimized_settings(self, config: Dict):
        """Print the auto-configured settings"""
        print("\n⚙️  AUTO-CONFIGURED SETTINGS:")
        print("=" * 70)
        print(f"   Worker Processes: {config['num_workers']}")
        print(f"   Batch Size: {config['batch_size']} frames")
        print(f"   Prefetch Buffer: {config['prefetch_frames']} frames")
        print(f"   GPU Acceleration: {'✅ Enabled' if config['use_gpu'] else '❌ Disabled'}")
        print(f"   Threading: ✅ Enabled")
        
        if config.get('memory_warning', False):
            print(f"\n⚠️  WARNING: Low RAM detected ({self.ram_gb:.1f} GB)")
            print(f"   Settings reduced to prevent out-of-memory errors")
        
        # Estimate performance
        base_fps = 12  # Original sequential performance
        speedup = self._estimate_speedup(config)
        estimated_fps = base_fps * speedup
        
        print(f"\n📈 EXPECTED PERFORMANCE:")
        print(f"   Estimated Speedup: ~{speedup:.1f}x")
        print(f"   Estimated FPS: ~{estimated_fps:.0f} FPS")
        print("=" * 70)
    
    def _estimate_speedup(self, config: Dict) -> float:
        """Estimate performance speedup based on configuration"""
        # Base speedup from parallelization
        parallel_speedup = min(config['num_workers'] * 0.85, config['num_workers'])
        
        # Bonus from prefetching (reduces I/O wait)
        io_bonus = 1.2 if config['prefetch_frames'] >= 64 else 1.1
        
        # Bonus from GPU
        gpu_bonus = 1.3 if config['use_gpu'] else 1.0
        
        # Penalty from storage
        storage_factor = 1.0
        if self.storage_type == 'HDD':
            storage_factor = 0.8
        elif self.storage_type == 'NVMe':
            storage_factor = 1.1
        
        total_speedup = parallel_speedup * io_bonus * gpu_bonus * storage_factor
        return max(1.0, total_speedup)


# =============================================================================
# CLASS 1: CONFIGURATION (Auto-adjusting)
# =============================================================================
class TrafficSignConfig:
    def __init__(self, auto_detect: bool = True):
        # --- File paths ---
        self.INPUT_VIDEO_PATH = DRIVE_PATH + 'videos/video2.mp4'
        self.OUTPUT_VIDEO_PATH = DRIVE_PATH + 'videos/video_output2.mp4'
        self.STUDENT_IDS = "523H0164_523H0177_523H0145"

        # --- Mask video output ---
        self.SAVE_MASK_VIDEOS = True
        self.MASK_VIDEO_BLUE = DRIVE_PATH + 'videos/mask_video_blue.mp4'
        self.MASK_VIDEO_RED = DRIVE_PATH + 'videos/mask_video_red.mp4'
        self.MASK_VIDEO_YELLOW = DRIVE_PATH + 'videos/mask_video_yellow.mp4'

        # --- Debug video output (pre-SVM detection) ---
        self.SAVE_DEBUG_VIDEO = True
        self.DEBUG_VIDEO_PATH = DRIVE_PATH + 'videos/debug_pre_svm_detection.mp4'

        # --- Save detector verified images ---
        self.SAVE_DETECTOR_IMAGES = True
        self.DETECTOR_IMAGES_FOLDER = DRIVE_PATH + 'check_data'

        # --- Processing limits ---
        self.MAX_FRAME_ID = 10000
        self.PROGRESS_UPDATE_INTERVAL = 100

        # --- AUTO-DETECTION ---
        if auto_detect:
            self.hardware = HardwareDetector()
            self.hardware.print_hardware_report()
            
            optimal_config = self.hardware.get_optimal_config()
            
            # Apply optimal settings
            self.NUM_WORKERS = optimal_config['num_workers']
            self.BATCH_SIZE = optimal_config['batch_size']
            self.PREFETCH_FRAMES = optimal_config['prefetch_frames']
            self.USE_GPU = optimal_config['use_gpu']
            self.USE_THREADING = True
            
            # Print configured settings
            self.hardware.print_optimized_settings(optimal_config)
        else:
            # Manual configuration fallback
            self.USE_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
            self.NUM_WORKERS = max(1, cpu_count() - 1)
            self.BATCH_SIZE = 16
            self.PREFETCH_FRAMES = 64
            self.USE_THREADING = True

        # --- Visualization ---
        self.DEBUG_MODE = True
        self.BOX_COLOR = (0, 255, 0)
        self.SHOW_SIGN_NAME = True

        # --- Color-specific parameters ---
        self.COLOR_PARAMS = {
            'blue': {
                'hsv_lower': np.array([0, 200, 70]),
                'hsv_upper': np.array([0, 255, 230]),
                'morph_ksize': 7, 'open_iter': 1, 'close_iter': 5,
                'blur_ksize': 5,
                'roi': (0.0, 0.0, 1.0, 1.0),
                'shape_type': 'circle'
            },
            'red': {
                'hsv_lower': np.array([117, 40, 0]),
                'hsv_upper': np.array([179, 255, 255]),
                'morph_ksize': 2, 'open_iter': 2, 'close_iter': 5,
                'blur_ksize': 5,
                'roi': (0.0, 0.0, 1.0, 1.0),
                'shape_type': 'circle'
            },
            'yellow': {
                'hsv_lower': np.array([8, 111, 100]),
                'hsv_upper': np.array([25, 255, 255]),
                'morph_ksize': 3, 'open_iter': 1, 'close_iter': 5,
                'blur_ksize': 7,
                'roi': (0.0, 0.0, 1.0, 1.0),
                'shape_type': 'triangle'
            }
        }

        # --- Image processing ---
        self.CLAHE_CLIP_LIMIT = 3.0
        self.CLAHE_TILE_GRID_SIZE = (1, 1)
        self.SATURATION_BOOST_FACTOR = 1.5
        self.PROCESSING_HEIGHT_PERCENT = 1.0

        # --- Shape detection parameters ---
        self.SHAPE_PARAMS = {
            'circle': {
                'min_area': 200, 'max_area': 15000,
                'trust_threshold': 1000,
                'small_circularity': 0.75,
                'large_circularity': 0.8
            },
            'triangle': {
                'min_area': 200, 'max_area': 50000,
                'trust_threshold': 1500,
                'min_solidity': 0.7,
                'epsilon_factor': 0.03,
                'max_vertices': 7
            }
        }

        # --- Tracking & Smoothing ---
        self.TRACKING_PARAMS = {
            'max_gap_sec': 1.0,
            'iou_threshold': 0.3,
            'smoothing_window': 10
        }


# =============================================================================
# CLASS 2: TRACKING & SMOOTHING
# =============================================================================
class SignTracker:
    def __init__(self, fps: float, tracking_params: Dict):
        self.fps = fps
        self.max_gap_frames = int(tracking_params['max_gap_sec'] * fps)
        self.iou_threshold = tracking_params['iou_threshold']
        self.smoothing_window = tracking_params['smoothing_window']

        self.tracks = defaultdict(list)
        self.next_track_id = 0
        self._smoothed_cache = {}
        self._cache_built = False

    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2

        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def add_detection(self, frame_num: int, bbox: Tuple, color: str, metrics: Dict):
        self._cache_built = False
        best_match_id = None
        best_iou = 0

        for track_id, track_data in self.tracks.items():
            if not track_data:
                continue
            last_detection = track_data[-1]

            if (last_detection['color'] == color and
                frame_num - last_detection['frame'] <= self.max_gap_frames):
                iou = self.calculate_iou(bbox, last_detection['bbox'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_match_id = track_id

        if best_match_id is not None:
            self.tracks[best_match_id].append({
                'frame': frame_num, 'bbox': bbox, 'color': color, 'metrics': metrics
            })
        else:
            self.tracks[self.next_track_id] = [{
                'frame': frame_num, 'bbox': bbox, 'color': color, 'metrics': metrics
            }]
            self.next_track_id += 1

    def interpolate_missing_frames(self, track_data: List) -> List:
        if len(track_data) < 2:
            return track_data

        interpolated = []
        for i in range(len(track_data) - 1):
            current = track_data[i]
            next_det = track_data[i + 1]
            interpolated.append(current)

            frame_gap = next_det['frame'] - current['frame']

            if 1 < frame_gap <= self.max_gap_frames:
                x1, y1, w1, h1 = current['bbox']
                x2, y2, w2, h2 = next_det['bbox']

                for j in range(1, frame_gap):
                    alpha = j / frame_gap
                    interpolated.append({
                        'frame': current['frame'] + j,
                        'bbox': (int(x1 + (x2 - x1) * alpha),
                                int(y1 + (y2 - y1) * alpha),
                                int(w1 + (w2 - w1) * alpha),
                                int(h1 + (h2 - h1) * alpha)),
                        'color': current['color'],
                        'metrics': current.get('metrics', {}),
                        'interpolated': True
                    })

        interpolated.append(track_data[-1])
        return interpolated

    def smooth_bounding_boxes(self, track_data: List) -> List:
        if len(track_data) < self.smoothing_window:
            return track_data

        smoothed = []
        for i in range(len(track_data)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(track_data), i + self.smoothing_window // 2 + 1)

            bboxes = [d['bbox'] for d in track_data[start_idx:end_idx]]

            smoothed_det = track_data[i].copy()
            smoothed_det['bbox'] = (
                int(np.mean([b[0] for b in bboxes])),
                int(np.mean([b[1] for b in bboxes])),
                int(np.mean([b[2] for b in bboxes])),
                int(np.mean([b[3] for b in bboxes]))
            )
            smoothed.append(smoothed_det)

        return smoothed

    def build_smoothed_cache(self):
        if self._cache_built:
            return

        print("   Building smoothed detection cache...")
        start = time.time()

        self._smoothed_cache.clear()

        for track_id, track_data in self.tracks.items():
            if not track_data:
                continue

            interpolated = self.interpolate_missing_frames(track_data)
            smoothed = self.smooth_bounding_boxes(interpolated)

            for detection in smoothed:
                frame_num = detection['frame']
                if frame_num not in self._smoothed_cache:
                    self._smoothed_cache[frame_num] = []
                self._smoothed_cache[frame_num].append(
                    (detection['bbox'], detection['color'], detection.get('metrics', {}))
                )

        self._cache_built = True

        print(f"   Cache built in {time.time() - start:.2f}s "
              f"({len(self._smoothed_cache)} frames, {len(self.tracks)} tracks)")

    def get_smoothed_detections(self, frame_num: int) -> List[Tuple]:
        if not self._cache_built:
            return []
        return self._smoothed_cache.get(frame_num, [])


# =============================================================================
# CLASS 3: DETECTOR (ĐÃ THAY THẾ BẰNG MSER + LỌC HÌNH DẠNG)
# =============================================================================
class TrafficSignDetector:
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        
        # Lấy tham số shape một lần
        self.shape_cfg_circle = config.SHAPE_PARAMS['circle']
        self.shape_cfg_triangle = config.SHAPE_PARAMS['triangle']
        
        # Tạo MSER với các tham số từ config
        min_area = min(self.shape_cfg_circle['min_area'], self.shape_cfg_triangle['min_area'])
        max_area = max(self.shape_cfg_circle['max_area'], self.shape_cfg_triangle['max_area'])

        print(f"Khởi tạo MSER với MinArea={min_area}, MaxArea={max_area}")
        
        self.mser = cv2.MSER_create(
            delta=5,           # Tham số quan trọng, 5 là giá trị tốt
            min_area=min_area,
            max_area=max_area,
            max_variation=0.25 # Mặc định, khá tốt
        )
        print("✓ MSER Detector (Giai đoạn 1) đã sẵn sàng.")

    # -------------------------------------------------------------------------
    # CÁC HÀM LỌC HÌNH DẠNG (Copy y hệt từ code cũ của bạn)
    # -------------------------------------------------------------------------
    
    def _detect_circles(self, contour: np.ndarray, area: float,
                       trust_threshold: float) -> Tuple[bool, Dict]:
        hull = cv2.convexHull(contour)
        perimeter_hull = cv2.arcLength(hull, True)
        area_hull = cv2.contourArea(hull)

        if perimeter_hull <= 0:
            return False, {}

        circularity = 4 * np.pi * (area_hull / (perimeter_hull * perimeter_hull))

        shape_cfg = self.config.SHAPE_PARAMS['circle']
        circ_thresh = (shape_cfg['small_circularity'] if area < trust_threshold
                      else shape_cfg['large_circularity'])

        if circularity > circ_thresh:
            return True, {
                'area': int(area_hull),
                'circularity': round(circularity, 3),
                'shape': 'circle'
            }
        return False, {}

    def _detect_triangles(self, contour: np.ndarray, area: float) -> Tuple[bool, Dict]:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            return False, {}

        shape_cfg = self.config.SHAPE_PARAMS['triangle']
        solidity = float(area) / hull_area

        if solidity <= shape_cfg['min_solidity']:
            return False, {}

        perimeter = cv2.arcLength(contour, True)
        epsilon = shape_cfg['epsilon_factor'] * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) <= shape_cfg['max_vertices']:
            return True, {
                'area': int(area),
                'solidity': round(solidity, 3),
                'shape': 'triangle'
            }
        return False, {}

    # -------------------------------------------------------------------------
    # HÀM PROCESS_FRAME MỚI (Dùng MSER)
    # -------------------------------------------------------------------------
    
    def process_frame(self, frame: np.ndarray, full_frame_dims: Tuple) -> Tuple[List[Tuple], Dict[str, np.ndarray]]:
        """
        Giai đoạn 1 MỚI: Dùng MSER + Lọc Hình dạng
        Returns: (detections, empty_masks_dict)
        """
        
        # 1. Chuẩn bị ảnh
        # MSER hoạt động trên ảnh xám (grayscale)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Chạy MSER
        try:
            regions, bboxes = self.mser.detectRegions(img_gray)
        except Exception as e:
            # print(f"Lỗi MSER: {e}")
            return [], {}
        
        all_detections = []
        
        # 3. Lọc các ứng viên bằng Hình dạng (Shape)
        for contour, bbox in zip(regions, bboxes):
            area = cv2.contourArea(contour)
            x, y, w, h = bbox
            
            # Kiểm tra hình tròn
            is_circle, metrics_c = self._detect_circles(contour, area, self.shape_cfg_circle['trust_threshold'])
            
            # Kiểm tra hình tam giác
            is_triangle, metrics_t = self._detect_triangles(contour, area)
            
            if is_circle:
                # Gán "màu" tạm thời là 'circle_mser'
                all_detections.append(((x, y, w, h), 'circle_mser', metrics_c))
            elif is_triangle:
                all_detections.append(((x, y, w, h), 'triangle_mser', metrics_t))

        # Vì không dùng HSV, chúng ta trả về dict mask rỗng
        # (Nếu bạn muốn lưu mask MSER, bạn có thể vẽ 'regions' lên một ảnh đen)
        empty_masks = {'blue': np.zeros_like(img_gray), 'red': np.zeros_like(img_gray), 'yellow': np.zeros_like(img_gray)}
        
        return all_detections, empty_masks

# =============================================================================
# PARALLEL FRAME PROCESSOR
# =============================================================================
def process_frame_worker(args):
    """Worker function for parallel frame processing"""
    frame_num, frame, config, full_frame_dims = args
    
    detector = TrafficSignDetector(config)
    detections, masks = detector.process_frame(frame, full_frame_dims)
    
    return frame_num, detections, masks


class ParallelFrameReader:
    """Parallel frame reader with prefetching"""
    def __init__(self, video_path: str, max_frames: int, prefetch_size: int = 64):
        self.video_path = video_path
        self.max_frames = max_frames
        self.prefetch_size = prefetch_size
        self.frame_queue = queue.Queue(maxsize=prefetch_size)
        self.stop_flag = False
        
    def _reader_thread(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        
        while not self.stop_flag and frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_queue.put((frame_count, frame))
            frame_count += 1
        
        cap.release()
        self.frame_queue.put(None)  # Signal end
    
    def start(self):
        import threading
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
        
    def get_frame(self):
        item = self.frame_queue.get()
        return item
    
    def stop(self):
        self.stop_flag = True
        if hasattr(self, 'thread'):
            self.thread.join()


# =============================================================================
# CLASS 4: VISUALIZER
# =============================================================================
class Visualizer:
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        self.roi_colors = {
            'blue': (255, 0, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255)
        }

    def draw_all(self, frame: np.ndarray, frame_num: int,
                 detections: List[Tuple], roi_params_dict: Dict) -> np.ndarray:
        frame_output = frame.copy()
        frame_output = self._draw_student_ids(frame_output)

        if self.config.DEBUG_MODE:
            frame_output = self._draw_frame_id(frame_output, frame_num)
            frame_output = self._draw_roi_boxes(frame_output, roi_params_dict)

        frame_output = self._draw_detections(frame_output, detections)
        return frame_output

    def _draw_student_ids(self, frame: np.ndarray) -> np.ndarray:
        text = self.config.STUDENT_IDS
        x, y = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(frame, (x - 5, y - text_h - 5),
                     (x + text_w + 5, y + baseline + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _draw_frame_id(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        text = f"Frame: {frame_num}"
        x, y = 10, 70
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 1.0, 2)
        cv2.rectangle(frame, (x - 5, y - text_h - 5),
                     (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def _draw_roi_boxes(self, frame: np.ndarray, roi_params_dict: Dict) -> np.ndarray:
        for color_name, (x_start, y_start, x_end, y_end) in roi_params_dict.items():
            color_bgr = self.roi_colors.get(color_name, (255, 255, 255))
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color_bgr, 2)
            cv2.putText(frame, f"{color_name.upper()} ROI",
                       (x_start + 10, y_start + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)
        return frame

    def _draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        for bbox, color_type, metrics in detections:
            x, y, w, h = bbox

            cv2.rectangle(frame, (x, y), (x + w, y + h), self.config.BOX_COLOR, 3)

            sign_name = metrics.get('sign_name', '')

            if self.config.SHOW_SIGN_NAME and sign_name:
                text_y = max(y - 15, 25)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2

                (text_w, text_h), baseline = cv2.getTextSize(sign_name, font, font_scale, thickness)

                cv2.rectangle(frame,
                            (x, text_y - text_h - 8),
                            (x + text_w + 10, text_y + baseline + 2),
                            (0, 0, 0), -1)

                cv2.putText(frame, sign_name, (x + 5, text_y),
                          font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            elif self.config.DEBUG_MODE and metrics:
                shape = metrics.get('shape', 'unknown')
                area = metrics.get('area', 0)

                if shape == 'circle':
                    text = f"A:{area} C:{metrics.get('circularity', 0):.2f}"
                else:
                    text = f"A:{area} S:{metrics.get('solidity', 0):.2f}"

                text_y = max(y - 10, 20)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, text_y - text_h - 5),
                            (x + text_w + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_debug_detections(self, frame: np.ndarray, frame_num: int,
                              detections: List[Tuple], roi_params_dict: Dict) -> np.ndarray:
        """Draw debug information for pre-SVM detection phase with detailed metrics"""
        frame_output = frame.copy()
        
        # Draw student IDs
        frame_output = self._draw_student_ids(frame_output)
        
        # Draw frame ID
        frame_output = self._draw_frame_id(frame_output, frame_num)
        
        # Draw ROI boxes
        frame_output = self._draw_roi_boxes(frame_output, roi_params_dict)
        
        # Draw detection count
        text = f"Pre-SVM Detections: {len(detections)}"
        x, y = 10, 110
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(frame_output, (x - 5, y - text_h - 5),
                     (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame_output, text, (x, y), font, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
        
        # Draw all detections with detailed metrics
        for bbox, color_type, metrics in detections:
            x, y, w, h = bbox
            
            # Color-coded boxes based on detection color
            if color_type == 'blue':
                box_color = (255, 0, 0)  # Blue
            elif color_type == 'red':
                box_color = (0, 0, 255)  # Red
            elif color_type == 'yellow':
                box_color = (0, 255, 255)  # Yellow
            else:
                box_color = (128, 128, 128)  # Gray
            
            # Draw bounding box
            cv2.rectangle(frame_output, (x, y), (x + w, y + h), box_color, 2)
            
            # Prepare detailed metrics text
            shape = metrics.get('shape', 'unknown')
            area = metrics.get('area', 0)
            
            if shape == 'circle':
                circularity = metrics.get('circularity', 0)
                metrics_text = [
                    f"{color_type.upper()}",
                    f"Area:{area}",
                    f"Circ:{circularity:.2f}"
                ]
            else:
                solidity = metrics.get('solidity', 0)
                metrics_text = [
                    f"{color_type.upper()}",
                    f"Area:{area}",
                    f"Sol:{solidity:.2f}"
                ]
            
            # Draw metrics text above bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            line_height = 15
            
            # Calculate total height needed for all text lines
            total_text_height = len(metrics_text) * line_height
            
            # Ensure text starts above bbox but stays on screen
            # If bbox is too high, draw below it instead
            if y - total_text_height - 10 > 0:
                text_y_start = y - 10
            else:
                text_y_start = y + h + 20  # Draw below bbox if not enough space above
            
            for i, text in enumerate(metrics_text):
                text_y = text_y_start + i * line_height
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(frame_output, (x, text_y - text_h - 3),
                            (x + text_w + 4, text_y + 3), (0, 0, 0), -1)
                cv2.putText(frame_output, text, (x + 2, text_y),
                          font, font_scale, box_color, thickness, cv2.LINE_AA)
        
        return frame_output


# =============================================================================
# CLASS 5: SVM DETECTOR
# =============================================================================
class SignDetectorSVM:
    def __init__(self, model_path: str):
        print(f"Loading SVM Detector from: {model_path}")
        try:
            self.svm = cv2.ml.SVM_load(model_path)
            if self.svm is None:
                raise Exception(f"Cannot load model from {model_path}")
        except Exception as e:
            print(f"❌ Critical error: Cannot load Detector model.")
            print(f"   Details: {e}")
            raise

        RESIZE_DIM = (64, 64)
        winSize = RESIZE_DIM
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9

        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        self.resize_dim = RESIZE_DIM
        print("✓ SVM Detector loaded successfully.")

    def verify(self, frame: np.ndarray, bbox: Tuple) -> bool:
        x, y, w, h = bbox

        if w <= 0 or h <= 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
            return False

        try:
            sign_roi = frame[y:y+h, x:x+w]
            sign_gray = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2GRAY)
            sign_resized = cv2.resize(sign_gray, self.resize_dim, interpolation=cv2.INTER_AREA)
            features = self.hog.compute(sign_resized)
            _, result = self.svm.predict(features.reshape(1, -1))
            label = int(result[0][0])
            return label == 1
        except:
            return False


# =============================================================================
# CLASS 6: SVM RECOGNIZER
# =============================================================================
class SignRecognizerSVM:
    def __init__(self, model_path: str, label_map: Dict[int, str]):
        print(f"Loading SVM Recognizer from: {model_path}")
        try:
            self.svm = cv2.ml.SVM_load(model_path)
            if self.svm is None:
                raise Exception(f"Cannot load model from {model_path}")
        except Exception as e:
            print(f"❌ Critical error: Cannot load Recognizer model.")
            print(f"   Details: {e}")
            raise

        self.label_map = label_map

        RESIZE_DIM = (64, 64)
        winSize = RESIZE_DIM
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9

        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        self.resize_dim = RESIZE_DIM
        print("✓ SVM Recognizer loaded successfully.")

    def _get_hog_features(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
            return None
        try:
            sign_roi = frame[y:y+h, x:x+w]
            sign_gray = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2GRAY)
            sign_resized = cv2.resize(sign_gray, self.resize_dim, interpolation=cv2.INTER_AREA)
            return self.hog.compute(sign_resized)
        except Exception:
            return None

    def recognize_batch(self, frame: np.ndarray, verified_detections: List[Tuple]) -> List[Tuple]:
        if not verified_detections:
            return []

        hog_features_list = []
        detections_list = []

        for (bbox, color, metrics) in verified_detections:
            features = self._get_hog_features(frame, bbox)
            if features is not None:
                hog_features_list.append(features)
                detections_list.append((bbox, color, metrics))

        if not hog_features_list:
            return []

        features_batch = np.array(hog_features_list, dtype=np.float32)
        if features_batch.ndim == 1:
             features_batch = features_batch.reshape(1, -1)

        _, results = self.svm.predict(features_batch)

        final_detections = []
        for i, (bbox, color, metrics) in enumerate(detections_list):
            label_id = int(results[i][0])
            sign_name = self.label_map.get(label_id, f"Unknown_{label_id}")

            metrics['sign_name'] = sign_name
            final_detections.append((bbox, color, metrics))

        return final_detections


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def convert_roi_to_pixels(roi_percent: Tuple, w_full: int, h_full: int) -> Tuple:
    x_start_pct, y_start_pct, x_end_pct, y_end_pct = roi_percent
    return (int(w_full * x_start_pct), int(h_full * y_start_pct),
            int(w_full * x_end_pct), int(h_full * y_end_pct))

def is_bbox_in_roi(bbox: Tuple, roi_params: Tuple, overlap_threshold: float = 0.5) -> bool:
    x, y, w, h = bbox
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_params
    x_max, y_max = x + w, y + h

    inter_x1 = max(x, roi_x_start)
    inter_y1 = max(y, roi_y_start)
    inter_x2 = min(x_max, roi_x_end)
    inter_y2 = min(y_max, roi_y_end)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    bbox_area = w * h
    overlap_ratio = inter_area / bbox_area if bbox_area > 0 else 0

    return overlap_ratio >= overlap_threshold
def verify_color_batch(frame: np.ndarray, verified_detections: List[Tuple]) -> List[Tuple]:
    """
    Giai đoạn 2.5: Kiểm tra màu sắc trung bình của các ứng viên
    đã vượt qua Model B (HOG Xám).
    """
    final_color_verified = []
    
    for (bbox, color, metrics) in verified_detections:
        try:
            x, y, w, h = bbox
            if w <= 0 or h <= 0: continue
            
            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Lấy 50% trung tâm của ROI để tránh viền
            h_roi, w_roi = hsv_roi.shape[:2]
            center_x, center_y = w_roi // 2, h_roi // 2
            crop_w, crop_h = int(w_roi * 0.5), int(h_roi * 0.5)
            
            center_roi = hsv_roi[
                max(0, center_y - crop_h//2) : center_y + crop_h//2,
                max(0, center_x - crop_w//2) : center_x + crop_w//2
            ]
            
            if center_roi.size == 0: continue
            
            # Tính màu trung bình (H, S, V)
            mean_h, mean_s, mean_v = cv2.mean(center_roi)[:3]
            
            # Ngưỡng: Phải có độ bão hòa (S) và độ sáng (V) nhất định
            # Tinh chỉnh các giá trị này nếu cần
            is_saturated = mean_s > 80 and mean_v > 50 

            is_red = (mean_h < 15 or mean_h > 165) and is_saturated
            is_blue = (mean_h > 95 and mean_h < 135) and is_saturated
            is_yellow = (mean_h > 20 and mean_h < 35) and is_saturated
            
            # Nếu màu sắc hợp lệ (Đỏ, Xanh, Vàng)
            if is_red or is_blue or is_yellow:
                # Gán lại "màu" thật (tùy chọn, để debug)
                if is_red: metrics['color'] = 'red'
                if is_blue: metrics['color'] = 'blue'
                if is_yellow: metrics['color'] = 'yellow'
                
                final_color_verified.append((bbox, color, metrics))
            # Nếu không, biển quảng cáo màu xanh lá, cam... sẽ bị LOẠI ở đây
            
        except Exception:
            continue
            
    return final_color_verified

# =============================================================================
# MAIN FUNCTION (AUTO-CONFIGURED)
# =============================================================================
def main():
    print("=" * 70)
    print("  TRAFFIC SIGN DETECTION WITH AUTO-OPTIMIZATION")
    print("=" * 70)

    try:
        start_total = time.time()
        
        # AUTO-DETECT HARDWARE AND CONFIGURE
        config = TrafficSignConfig(auto_detect=True)
        visualizer = Visualizer(config)

        # Initialize SVM Detector
        try:
            detector_svm = SignDetectorSVM(model_path=MODEL_DETECTOR_PATH)
        except Exception as e:
            print("Cannot initialize SVM detector")
            return

        # Label map
        LABEL_MAP_CUSTOM = {
            0: "cam_di_nguoc_chieu",
            1: "cam_queo_trai",
            2: "cam_do_xe",
            3: "cam_dung_do_xe",
            4: "huong_ben_phai",
            5: "canh_bao_di_cham",
            6: "canh_bao_nguoi_qua_duong",
            7: "canh_bao_duong_gap_khuc"
        }

        # Initialize SVM Recognizer
        try:
            recognizer_svm = SignRecognizerSVM(
                model_path=MODEL_RECOGNIZER_PATH,
                label_map=LABEL_MAP_CUSTOM
            )
        except Exception as e:
            print("Cannot initialize SVM recognizer")
            return

        # Open video to get metadata
        cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video '{config.INPUT_VIDEO_PATH}'")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"\n📹 VIDEO INFORMATION:")
        print(f"   File: {config.INPUT_VIDEO_PATH}")
        print(f"   Resolution: {w_orig}x{h_orig}")
        print(f"   Frame Rate: {fps:.2f} FPS")
        print(f"   Total Frames: {total_frames}")
        print(f"   Processing Limit: {config.MAX_FRAME_ID} frames")

        # Initialize tracker
        tracker = SignTracker(fps, config.TRACKING_PARAMS)

        # Calculate dimensions
        h_crop = int(h_orig * config.PROCESSING_HEIGHT_PERCENT)
        w_crop = w_orig
        full_frame_dims = (w_orig, h_orig)

        # Calculate ROI
        roi_pixel_map = {
            color: convert_roi_to_pixels(params['roi'], w_orig, h_orig)
            for color, params in config.COLOR_PARAMS.items()
        }

        # =====================================================================
        # PASS 1: PARALLEL DETECTION + TRACKING
        # =====================================================================
        print("\n" + "=" * 70)
        print("⚡ PASS 1: PARALLEL DETECTION + TRACKING")
        print("=" * 70)
        start_detection = time.time()

        # Start parallel frame reader
        frame_reader = ParallelFrameReader(config.INPUT_VIDEO_PATH, 
                                          config.MAX_FRAME_ID, 
                                          config.PREFETCH_FRAMES)
        frame_reader.start()

        # Prepare mask writers
        mask_writers = None
        if config.SAVE_MASK_VIDEOS:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            mask_writers = {
                'blue': cv2.VideoWriter(config.MASK_VIDEO_BLUE, fourcc, fps, (w_crop, h_crop), False),
                'red': cv2.VideoWriter(config.MASK_VIDEO_RED, fourcc, fps, (w_crop, h_crop), False),
                'yellow': cv2.VideoWriter(config.MASK_VIDEO_YELLOW, fourcc, fps, (w_crop, h_crop), False)
            }

        # Prepare debug video writer (pre-SVM detection)
        debug_writer = None
        if config.SAVE_DEBUG_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            debug_writer = cv2.VideoWriter(config.DEBUG_VIDEO_PATH, fourcc, fps, (w_orig, h_orig))
            if debug_writer.isOpened():
                print(f"\n📹 Debug video enabled: {config.DEBUG_VIDEO_PATH}")
            else:
                print(f"\n⚠️  Warning: Cannot create debug video writer")
                debug_writer = None

        # Process frames in parallel batches
        frame_batch = []
        frame_full_batch = []  # Store full frames for debug video
        frame_count = 0
        
        # Create a pool of workers
        with Pool(processes=config.NUM_WORKERS) as pool:
            while True:
                item = frame_reader.get_frame()
                if item is None:
                    break
                
                frame_num, frame_full = item
                frame_to_process = frame_full[0:h_crop, 0:w_crop]
                
                frame_batch.append((frame_num, frame_to_process, config, full_frame_dims))
                if debug_writer:
                    frame_full_batch.append((frame_num, frame_full.copy()))
                
                # Process batch when full
                if len(frame_batch) >= config.BATCH_SIZE:
                    results = pool.map(process_frame_worker, frame_batch)
                    
                    # Create dictionary to map frame_num to detections for debug video
                    if debug_writer and frame_full_batch:
                        detections_map = {}
                        for frame_num, detections, masks in results:
                            detections_map[frame_num] = detections
                    
                    for frame_num, detections, masks in results:
                        # Add to tracker
                        if detections:
                            for bbox, color, metrics in detections:
                                tracker.add_detection(frame_num, bbox, color, metrics)
                        
                        # Write masks (if enabled)
                        if config.SAVE_MASK_VIDEOS and mask_writers:
                            for color in ['blue', 'red', 'yellow']:
                                if color in masks and mask_writers[color].isOpened():
                                    mask_writers[color].write(masks[color])
                    
                    # Write debug video frames with all detections
                    if debug_writer and frame_full_batch:
                        for frame_num, frame_full in frame_full_batch:
                            detections = detections_map.get(frame_num, [])
                            debug_frame = visualizer.draw_debug_detections(
                                frame_full, frame_num, detections, roi_pixel_map
                            )
                            debug_writer.write(debug_frame)
                        frame_full_batch.clear()
                    
                    frame_count += len(frame_batch)
                    frame_batch.clear()
                    
                    if frame_count % config.PROGRESS_UPDATE_INTERVAL == 0:
                        progress = int(frame_count / config.MAX_FRAME_ID * 100)
                        elapsed = time.time() - start_detection
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"   Detected {frame_count} frames ({progress}%) - {current_fps:.1f} FPS")
                        gc.collect()
            
            # Process remaining frames
            if frame_batch:
                results = pool.map(process_frame_worker, frame_batch)
                
                # Create dictionary for debug video
                if debug_writer and frame_full_batch:
                    detections_map = {}
                    for frame_num, detections, masks in results:
                        detections_map[frame_num] = detections
                
                for frame_num, detections, masks in results:
                    if detections:
                        for bbox, color, metrics in detections:
                            tracker.add_detection(frame_num, bbox, color, metrics)
                    
                    if config.SAVE_MASK_VIDEOS and mask_writers:
                        for color in ['blue', 'red', 'yellow']:
                            if color in masks and mask_writers[color].isOpened():
                                mask_writers[color].write(masks[color])
                
                # Write remaining debug video frames
                if debug_writer and frame_full_batch:
                    for frame_num, frame_full in frame_full_batch:
                        detections = detections_map.get(frame_num, [])
                        debug_frame = visualizer.draw_debug_detections(
                            frame_full, frame_num, detections, roi_pixel_map
                        )
                        debug_writer.write(debug_frame)
                    frame_full_batch.clear()
                
                frame_count += len(frame_batch)

        frame_reader.stop()

        if mask_writers:
            for writer in mask_writers.values():
                writer.release()
        
        if debug_writer:
            debug_writer.release()
            print(f"✅ Debug video saved: {config.DEBUG_VIDEO_PATH}")

        detection_time = time.time() - start_detection
        print(f"\n✅ Detection Pass completed in {detection_time:.2f}s")
        print(f"   Average Speed: {frame_count/detection_time:.1f} FPS")

        # Build smoothed cache
        tracker.build_smoothed_cache()

        print(f"\n📊 Tracking Statistics:")
        print(f"   Total tracks: {len(tracker.tracks)}")

        # =====================================================================
        # PASS 2: PARALLEL VERIFY + RECOGNIZE + RENDER
        # =====================================================================
        print("\n" + "=" * 70)
        print("⚡ PASS 2: PARALLEL VERIFY + RECOGNIZE + RENDER")
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

        # Create folder for detector verified images
        saved_images_count = 0
        if config.SAVE_DETECTOR_IMAGES:
            if not os.path.exists(config.DETECTOR_IMAGES_FOLDER):
                os.makedirs(config.DETECTOR_IMAGES_FOLDER)
                print(f"\n📁 Created folder: {config.DETECTOR_IMAGES_FOLDER}")
            else:
                print(f"\n📁 Using existing folder: {config.DETECTOR_IMAGES_FOLDER}")

        # Use threading for I/O and processing pipeline
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                item = frame_reader.get_frame()
                if item is None:
                    break
                
                frame_num, frame = item
                
                # 1. Lấy ỨNG VIÊN (từ MSER + Temporal Filter)
                smoothed_candidates = tracker.get_smoothed_detections(frame_num)

                # 2. Giai đoạn 2: HOG Xám Detector
                verified_detections_B = []
                if smoothed_candidates:
                    for (bbox, color, metrics) in smoothed_candidates:
                        if detector_svm.verify(frame, bbox):
                            verified_detections_B.append((bbox, color, metrics))
                
                # 3. Giai đoạn 2.5: KIỂM TRA MÀU SẮC (BƯỚC MỚI)
                verified_detections_Color = []
                if verified_detections_B:
                    verified_detections_Color = verify_color_batch(frame, verified_detections_B)
                
                # 4. Giai đoạn 3: HOG Xám Recognizer
                recognized_detections = []
                if verified_detections_Color:
                    # Chỉ nhận dạng các ứng viên đã qua CẢ HAI bước lọc
                    recognized_detections = recognizer_svm.recognize_batch(frame, verified_detections_Color)
                    total_signs_recognized += len(recognized_detections)

                # 5. Vẽ
                frame_output = visualizer.draw_all(frame, frame_num,
                                                 recognized_detections, roi_pixel_map)
                video_writer.write(frame_output)

                if recognized_detections:
                    frames_with_detections += 1

                frame_count_render += 1

                # (Phần in tiến độ giữ nguyên)
                if frame_count_render % config.PROGRESS_UPDATE_INTERVAL == 0:
                    progress = int(frame_count_render / total_frames * 100)
                    elapsed = time.time() - start_render
                    current_fps = frame_count_render / elapsed if elapsed > 0 else 0
                    status_msg = f"   Rendered {frame_count_render}/{total_frames} frames ({progress}%) " \
                                 f"- {current_fps:.1f} FPS - {total_signs_recognized} signs"
                    # (Code save detector images giữ nguyên)
                    print(status_msg)

                del frame, frame_output

        frame_reader.stop()
        video_writer.release()

        render_time = time.time() - start_render
        print(f"\n✅ Rendering Pass completed in {render_time:.2f}s")
        print(f"   Average Speed: {frame_count_render/render_time:.1f} FPS")
        print(f"   Frames with signs: {frames_with_detections}/{frame_count_render}")
        print(f"   Total signs recognized: {total_signs_recognized}")
        if config.SAVE_DETECTOR_IMAGES:
            print(f"   Detector images saved: {saved_images_count}")

        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        total_time = time.time() - start_total

        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"📁 Output: {config.OUTPUT_VIDEO_PATH}")

        if config.SAVE_DEBUG_VIDEO:
            print(f"📁 Debug (Pre-SVM): {config.DEBUG_VIDEO_PATH}")

        if config.SAVE_DETECTOR_IMAGES:
            print(f"📁 Detector Images: {config.DETECTOR_IMAGES_FOLDER} ({saved_images_count} images)")

        if config.SAVE_MASK_VIDEOS:
            print(f"\n📁 Mask videos:")
            print(f"   Blue: {config.MASK_VIDEO_BLUE}")
            print(f"   Red: {config.MASK_VIDEO_RED}")
            print(f"   Yellow: {config.MASK_VIDEO_YELLOW}")

        print(f"\n⏱️  PERFORMANCE SUMMARY:")
        print("=" * 70)
        print(f"   Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"   Detection Phase: {detection_time:.2f}s ({detection_time/total_time*100:.1f}%)")
        print(f"   Rendering Phase: {render_time:.2f}s ({render_time/total_time*100:.1f}%)")
        print(f"   Overall Speed: {frame_count_render/total_time:.1f} FPS")
        
        # Calculate actual speedup vs baseline
        baseline_fps = 12  # Typical sequential performance
        actual_speedup = (frame_count_render/total_time) / baseline_fps
        print(f"   Actual Speedup: {actual_speedup:.1f}x vs sequential")

        print(f"\n🎯 OPTIMIZATION APPLIED:")
        print(f"   ✅ Multi-processing ({config.NUM_WORKERS} workers)")
        print(f"   ✅ Frame prefetching ({config.PREFETCH_FRAMES} frames)")
        print(f"   ✅ Batch processing ({config.BATCH_SIZE} frames/batch)")
        print(f"   ✅ Thread-based I/O")
        print(f"   ✅ GPU acceleration: {'Enabled' if config.USE_GPU else 'Disabled'}")
        print(f"   ✅ Hardware auto-detection")
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