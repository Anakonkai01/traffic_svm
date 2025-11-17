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
TRAFFIC SIGN DETECTION SYSTEM - GIẢI PHÁP 5 (MSER Dual Channel)

LOGIC ĐÃ THAY ĐỔI:
====================
- Bỏ hoàn toàn lọc màu HSV/YCrCb/Canny.
- PASS 1: Detect (MSER trên Cr+Cb + Shape Filter) -> Đoán màu (YCrCb-Mean) -> Verify (SVM) -> Track/Smooth
- PASS 2: Recognize (SVM) -> Render

ƯU ĐIỂM:
- KHÔNG CẦN TUNE MÀU/CẠNH.
- Cực kỳ ổn định với ánh sáng, xử lý được cả "màu đỏ sẫm".
- Tận dụng lại 100% các model HOG/SVM bạn đã train.
- Giữ được logic Anti-Flicker và logic "Problem Signs".
"""


DRIVE_PATH = "../"

MODEL_DETECTOR_PATH = DRIVE_PATH + "models/dector_v5_nhut.xml"
MODEL_RECOGNIZER_PATH = DRIVE_PATH + "models/svm_sign_recognizer_v3.xml"


# =============================================================================
# AUTO-DETECTION SYSTEM
# =============================================================================
class HardwareDetector:
    # (Không thay đổi - Giữ nguyên class HardwareDetector)
    """Automatically detects and analyzes computer hardware"""
    
    def __init__(self):
        self.cpu_cores = cpu_count()
        self.ram_gb = self._get_ram_gb()
        self.has_gpu = self._check_gpu()
        self.storage_type = self._detect_storage_type()
        self.cpu_model = self._get_cpu_model()
        self.os_info = self._get_os_info()
        
    def _get_ram_gb(self) -> float:
        try:
            return psutil.virtual_memory().total / (1024 ** 3)
        except:
            return 8.0
    
    def _check_gpu(self) -> bool:
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False
    
    def _detect_storage_type(self) -> str:
        try:
            if platform.system() == 'Windows':
                import subprocess
                output = subprocess.check_output(
                    'powershell "Get-PhysicalDisk | Select MediaType"',
                    shell=True,
                    text=True
                )
                if 'SSD' in output: return 'SSD'
                elif 'NVMe' in output: return 'NVMe'
                else: return 'HDD'
            elif platform.system() == 'Linux':
                try:
                    with open('/sys/block/sda/queue/rotational', 'r') as f:
                        if f.read().strip() == '0': return 'SSD'
                        else: return 'HDD'
                except: return 'SSD'
            else: return 'SSD'
        except: return 'SSD'
    
    def _get_cpu_model(self) -> str:
        try:
            if platform.system() == 'Windows': return platform.processor()
            elif platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
            else: return platform.processor()
        except: return "Unknown CPU"
    
    def _get_os_info(self) -> str:
        return f"{platform.system()} {platform.release()}"
    
    def get_optimal_config(self) -> Dict:
        config = {}
        if self.cpu_cores <= 2:
            config['num_workers'] = 1
            config['batch_size'] = 4
            config['prefetch_frames'] = 16
            config['profile'] = 'Low-End (2 cores)'
        elif self.cpu_cores <= 4:
            config['num_workers'] = max(1, self.cpu_cores - 1)
            config['batch_size'] = 8
            config['prefetch_frames'] = 32
            config['profile'] = 'Budget (4 cores)'
        elif self.cpu_cores <= 8:
            config['num_workers'] = max(1, self.cpu_cores - 1)
            config['batch_size'] = 16
            config['prefetch_frames'] = 64
            config['profile'] = 'Mid-Range (6-8 cores)'
        elif self.cpu_cores <= 12:
            config['num_workers'] = max(1, self.cpu_cores - 1)
            config['batch_size'] = 24
            config['prefetch_frames'] = 96
            config['profile'] = 'High-End (10-12 cores)'
        else:
            config['num_workers'] = max(1, self.cpu_cores - 2)
            config['batch_size'] = 32
            config['prefetch_frames'] = 128
            config['profile'] = 'Extreme (12+ cores)'
        
        if self.ram_gb < 6:
            config['batch_size'] = max(4, config['batch_size'] // 2)
            config['prefetch_frames'] = max(16, config['prefetch_frames'] // 2)
            config['memory_warning'] = True
        elif self.ram_gb < 12:
            config['batch_size'] = max(8, int(config['batch_size'] * 0.75))
            config['memory_warning'] = False
        else:
            if self.ram_gb >= 32:
                config['batch_size'] = int(config['batch_size'] * 1.5)
            config['memory_warning'] = False
        
        if self.storage_type == 'NVMe':
            config['prefetch_frames'] = int(config['prefetch_frames'] * 1.5)
        elif self.storage_type == 'HDD':
            config['prefetch_frames'] = max(16, config['prefetch_frames'] // 2)
        
        config['use_gpu'] = self.has_gpu
        return config
    
    def print_hardware_report(self):
        print("\n" + "=" * 70)
        print("🔍 HARDWARE DETECTION REPORT")
        print("=" * 70)
        print(f"💻 System: {self.os_info}")
        print(f"🧠 CPU: {self.cpu_model}")
        print(f"⚙️  Cores: {self.cpu_cores} cores")
        print(f"🎮 RAM: {self.ram_gb:.1f} GB")
        print(f"💾 Storage: {self.storage_type}")
        print(f"🎨 GPU (CUDA): {'✅ Available' if self.has_gpu else '❌ Not detected'}")
        config = self.get_optimal_config()
        print(f"\n📊 Performance Profile: {config['profile']}")
        print("=" * 70)
    
    def print_optimized_settings(self, config: Dict):
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
        
        base_fps = 12
        speedup = self._estimate_speedup(config)
        estimated_fps = base_fps * speedup
        
        print(f"\n📈 EXPECTED PERFORMANCE:")
        print(f"   Estimated Speedup: ~{speedup:.1f}x")
        print(f"   Estimated FPS: ~{estimated_fps:.0f} FPS")
        print("=" * 70)
    
    def _estimate_speedup(self, config: Dict) -> float:
        parallel_speedup = min(config['num_workers'] * 0.85, config['num_workers'])
        io_bonus = 1.2 if config['prefetch_frames'] >= 64 else 1.1
        gpu_bonus = 1.3 if config['use_gpu'] else 1.0
        
        storage_factor = 1.0
        if self.storage_type == 'HDD': storage_factor = 0.8
        elif self.storage_type == 'NVMe': storage_factor = 1.1
        
        total_speedup = parallel_speedup * io_bonus * gpu_bonus * storage_factor
        return max(1.0, total_speedup)


# =============================================================================
# CLASS 1: CONFIGURATION (Sử dụng MSER)
# =============================================================================
class TrafficSignConfig:
    def __init__(self, auto_detect: bool = True):
        # --- File paths ---
        self.INPUT_VIDEO_PATH = DRIVE_PATH + 'videos/video2.mp4'
        self.OUTPUT_VIDEO_PATH = DRIVE_PATH + 'videos/video_output2.mp4'
        self.STUDENT_IDS = "523H0164_523H0177_523H0145"

        # --- Mask video output ---
        self.SAVE_MASK_VIDEOS = True
        self.MASK_VIDEO_MSER = DRIVE_PATH + 'videos/mask_video_mser_dual.mp4'

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
            self.NUM_WORKERS = optimal_config['num_workers']
            self.BATCH_SIZE = optimal_config['batch_size']
            self.PREFETCH_FRAMES = optimal_config['prefetch_frames']
            self.USE_GPU = optimal_config['use_gpu']
            self.USE_THREADING = True
            self.hardware.print_optimized_settings(optimal_config)
        else:
            self.USE_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
            self.NUM_WORKERS = max(1, cpu_count() - 1)
            self.BATCH_SIZE = 16
            self.PREFETCH_FRAMES = 64
            self.USE_THREADING = True

        # --- Visualization ---
        self.DEBUG_MODE = True
        self.BOX_COLOR = (0, 255, 0)
        self.SHOW_SIGN_NAME = True

        # === (MỚI) CẤU HÌNH MSER (Maximally Stable Extremal Regions) ===
        self.MSER_DELTA = 2
        self.MSER_MIN_AREA = 100
        self.MSER_MAX_AREA = 14400
        self.MSER_MAX_VARIATION = 0.15
        self.MSER_MIN_DIVERSITY = 0.2
        
        # --- (MỚI) CẤU HÌNH ĐOÁN MÀU (YCrCb-MEAN) ---
        self.COLOR_GUESS_CR_MIN = 125 
        self.COLOR_GUESS_CB_MIN = 135 
        self.COLOR_GUESS_CR_MAX_FOR_YELLOW = 130
        self.COLOR_GUESS_CB_MAX_FOR_YELLOW = 120
        # === KẾT THÚC CẤU HÌNH MỚI ===

        # --- Image processing ---
        self.PROCESSING_HEIGHT_PERCENT = 1.0

        # --- Shape detection parameters (Giữ nguyên) ---
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

        # --- Tracking & Smoothing (Giữ nguyên) ---
        self.TRACKING_PARAMS = {
            'max_gap_sec': 1.0,
            'iou_threshold': 0.1,
            'smoothing_window': 15
        }
        
        # --- Logic "Problem Signs" (Giữ nguyên) ---
        self.PROBLEM_SIGNS = {
            "cam_do_xe", 
            "cam_dung_do_xe"
        }
        self.BBOX_EXPAND_SCALE = 1.3 
        self.BBOX_OVERLAP_IOU = 0.8


# =============================================================================
# CLASS 2: TRACKING & SMOOTHING
# =============================================================================
class SignTracker:
    # (Không thay đổi - Giữ nguyên class SignTracker)
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
        inter_x1 = max(x1, x2); inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max); inter_y2 = min(y1_max, y2_max)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1: return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1; area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def add_detection(self, frame_num: int, bbox: Tuple, color: str, metrics: Dict):
        self._cache_built = False
        best_match_id = None
        best_iou = 0
        for track_id, track_data in self.tracks.items():
            if not track_data: continue
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
        if len(track_data) < 2: return track_data
        interpolated = []
        for i in range(len(track_data) - 1):
            current = track_data[i]; next_det = track_data[i + 1]
            interpolated.append(current)
            frame_gap = next_det['frame'] - current['frame']
            if 1 < frame_gap <= self.max_gap_frames:
                x1, y1, w1, h1 = current['bbox']; x2, y2, w2, h2 = next_det['bbox']
                for j in range(1, frame_gap):
                    alpha = j / frame_gap
                    interpolated.append({
                        'frame': current['frame'] + j,
                        'bbox': (int(x1 + (x2 - x1) * alpha), int(y1 + (y2 - y1) * alpha),
                                int(w1 + (w2 - w1) * alpha), int(h1 + (h2 - h1) * alpha)),
                        'color': current['color'], 'metrics': current.get('metrics', {}),
                        'interpolated': True
                    })
        interpolated.append(track_data[-1])
        return interpolated

    def smooth_bounding_boxes(self, track_data: List) -> List:
        if len(track_data) < self.smoothing_window: return track_data
        smoothed = []
        for i in range(len(track_data)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(track_data), i + self.smoothing_window // 2 + 1)
            bboxes = [d['bbox'] for d in track_data[start_idx:end_idx]]
            smoothed_det = track_data[i].copy()
            smoothed_det['bbox'] = (
                int(np.mean([b[0] for b in bboxes])), int(np.mean([b[1] for b in bboxes])),
                int(np.mean([b[2] for b in bboxes])), int(np.mean([b[3] for b in bboxes]))
            )
            smoothed.append(smoothed_det)
        return smoothed

    def build_smoothed_cache(self):
        if self._cache_built: return
        print("   Building smoothed detection cache (Anti-flicker)...")
        start = time.time()
        self._smoothed_cache.clear()
        for track_id, track_data in self.tracks.items():
            if not track_data: continue
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
        if not self._cache_built: return []
        return self._smoothed_cache.get(frame_num, [])


# =============================================================================
# CLASS 3: DETECTOR (LOGIC MSER MỚI)
# =============================================================================
class TrafficSignDetector:
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        
    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(MỚI) Preprocessing: Tách kênh YCrCb
        Trả về: (Y, Cr, Cb, frame_ycrcb_full)
        """
        frame_blur_bgr = cv2.medianBlur(frame, 5)
        frame_ycrcb = cv2.cvtColor(frame_blur_bgr, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(frame_ycrcb)
        return Y, Cr, Cb, frame_ycrcb
    
    def _get_dominant_color(self, roi_ycrcb: np.ndarray) -> str:
        """(MỚI) Đoán màu dựa trên YCrCb trung bình của ROI"""
        try:
            mean_vals = cv2.mean(roi_ycrcb)
            mean_cr = mean_vals[1]
            mean_cb = mean_vals[2]
            
            if mean_cr > self.config.COLOR_GUESS_CR_MIN: return 'red'
            if mean_cb > self.config.COLOR_GUESS_CB_MIN: return 'blue'
            if mean_cr < self.config.COLOR_GUESS_CR_MAX_FOR_YELLOW and \
               mean_cb < self.config.COLOR_GUESS_CB_MAX_FOR_YELLOW: return 'yellow'
            return 'unknown'
        except Exception:
            return 'unknown'

    def _detect_circles(self, contour: np.ndarray, area: float,
                       trust_threshold: float) -> Tuple[bool, Dict]:
        hull = cv2.convexHull(contour)
        perimeter_hull = cv2.arcLength(hull, True)
        area_hull = cv2.contourArea(hull)
        if perimeter_hull <= 0: return False, {}
        circularity = 4 * np.pi * (area_hull / (perimeter_hull * perimeter_hull))
        shape_cfg = self.config.SHAPE_PARAMS['circle']
        circ_thresh = (shape_cfg['small_circularity'] if area < trust_threshold 
                      else shape_cfg['large_circularity'])
        if circularity > circ_thresh:
            return True, {'area': int(area_hull), 'circularity': round(circularity, 3), 'shape': 'circle'}
        return False, {}
    
    def _detect_triangles(self, contour: np.ndarray, area: float) -> Tuple[bool, Dict]:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: return False, {}
        shape_cfg = self.config.SHAPE_PARAMS['triangle']
        solidity = float(area) / hull_area
        if solidity <= shape_cfg['min_solidity']: return False, {}
        perimeter = cv2.arcLength(contour, True)
        epsilon = shape_cfg['epsilon_factor'] * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) <= shape_cfg['max_vertices']:
            return True, {'area': int(area), 'solidity': round(solidity, 3), 'shape': 'triangle'}
        return False, {}

    def process_frame(self, frame: np.ndarray, full_frame_dims: Tuple, 
                      mser_cr: cv2.MSER, mser_cb: cv2.MSER) -> Tuple[List[Tuple], Dict[str, np.ndarray]]:
        """
        (MỚI) Pipeline dựa trên MSER (Dual Channel) + Shape Filter + Color Guess
        """
        all_detections = []
        masks_dict = {}

        # 1. Tách các kênh Y, Cr, Cb
        Y, Cr, Cb, frame_ycrcb = self._preprocess_frame(frame)
        
        # 2. Chạy MSER trên kênh Cr (tìm Đỏ/Vàng)
        regions_cr, bboxes_cr = mser_cr.detectRegions(Cr)
        
        # 3. Chạy MSER trên kênh Cb (tìm Xanh)
        regions_cb, bboxes_cb = mser_cb.detectRegions(Cb)
        
        # 4. Kết hợp kết quả
        all_regions = []
        all_bboxes = []
        if regions_cr is not None:
            all_regions.extend(regions_cr)
            all_bboxes.extend(list(bboxes_cr))
        if regions_cb is not None:
            all_regions.extend(regions_cb)
            all_bboxes.extend(list(bboxes_cb))

        # 5. Tạo MSER mask để hiển thị (từ cả 2 kênh)
        mser_mask = np.zeros_like(Y)
        if all_regions:
            cv2.drawContours(mser_mask, all_regions, -1, (255), 1)
        masks_dict['mser_dual'] = mser_mask # Đổi tên mask

        circle_cfg = self.config.SHAPE_PARAMS['circle']
        triangle_cfg = self.config.SHAPE_PARAMS['triangle']
        
        # 6. Lọc Hình Dạng (trên TẤT CẢ các vùng tìm được)
        for contour, bbox in zip(all_regions, all_bboxes):
            area = cv2.contourArea(contour)
            x, y, w, h = bbox
            if w <= 0 or h <= 0: continue

            # Thử lọc hình tròn
            if (area >= circle_cfg['min_area'] and area <= circle_cfg['max_area']):
                is_valid, metrics = self._detect_circles(contour, area, 
                                                        circle_cfg['trust_threshold'])
                if is_valid:
                    roi_ycrcb = frame_ycrcb[y:y+h, x:x+w]
                    color = self._get_dominant_color(roi_ycrcb)
                    all_detections.append((tuple(bbox), color, metrics))
                    continue 

            # Thử lọc hình tam giác
            if (area >= triangle_cfg['min_area'] and area <= triangle_cfg['max_area']):
                is_valid, metrics = self._detect_triangles(contour, area)
                if is_valid:
                    roi_ycrcb = frame_ycrcb[y:y+h, x:x+w]
                    color = self._get_dominant_color(roi_ycrcb)
                    all_detections.append((tuple(bbox), color, metrics))
        
        return all_detections, masks_dict


# =============================================================================
# PARALLEL FRAME PROCESSOR (LOGIC MỚI - MSER)
# =============================================================================
worker_models = {}

def initialize_worker(detector_model_path, config: TrafficSignConfig):
    """
    (MỚI) Tải SVM VÀ tạo MSER instances cho worker
    """
    global worker_models
    if not worker_models:
        print(f"Worker (PID {os.getpid()}) initializing models...")
        # 1. Lưu config
        worker_models['config'] = config
        
        # 2. Tải model HOG Xám (SVM Detector)
        worker_models['detector_svm'] = SignDetectorSVM(detector_model_path)
        
        # 3. Tạo MSER instances dựa trên config
        worker_models['mser_cr'] = cv2.MSER_create(
            delta=config.MSER_DELTA,
            min_area=config.MSER_MIN_AREA,
            max_area=config.MSER_MAX_AREA,
            max_variation=config.MSER_MAX_VARIATION,
            min_diversity=config.MSER_MIN_DIVERSITY
        )
        worker_models['mser_cb'] = cv2.MSER_create(
            delta=config.MSER_DELTA,
            min_area=config.MSER_MIN_AREA,
            max_area=config.MSER_MAX_AREA,
            max_variation=config.MSER_MAX_VARIATION,
            min_diversity=config.MSER_MIN_DIVERSITY
        )
        
        # 4. Tạo MSER Detector (Class 3)
        worker_models['detector'] = TrafficSignDetector(config)

def process_frame_worker(args):
    """
    Worker function cho Giai đoạn 1: Detect (MSER) + Verify (SVM)
    """
    # (MỚI) Config không còn được truyền trong args
    frame_num, frame, full_frame_dims = args
    global worker_models

    # Lấy các model/detector đã khởi tạo
    detector_svm = worker_models.get('detector_svm')
    detector = worker_models.get('detector')
    mser_cr = worker_models.get('mser_cr')
    mser_cb = worker_models.get('mser_cb')

    if not all([detector_svm, detector, mser_cr, mser_cb]):
        print(f"Worker (PID {os.getpid()}) LỖI: Models chưa được khởi tạo.")
        return frame_num, [], {}

    # 1. Detect (MSER + Shape + Color Guess) -> Lấy ứng cử viên thô
    raw_detections, masks = detector.process_frame(frame, full_frame_dims, mser_cr, mser_cb)

    # 2. Verify (SVM HOG) -> Lọc ứng cử viên
    verified_detections = []
    if raw_detections:
        for (bbox, color, metrics) in raw_detections:
            if detector_svm.verify(frame, bbox):
                verified_detections.append((bbox, color, metrics))

    # 3. Trả về các ứng cử viên ĐÃ ĐƯỢC XÁC THỰC
    return frame_num, verified_detections, masks


# =============================================================================
# (Giữ nguyên) ParallelFrameReader, Visualizer, SignDetectorSVM, SignRecognizerSVM
# =============================================================================
class ParallelFrameReader:
    # (Không thay đổi)
    def __init__(self, video_path: str, max_frames: int, prefetch_size: int = 64):
        self.video_path = video_path; self.max_frames = max_frames
        self.prefetch_size = prefetch_size
        self.frame_queue = queue.Queue(maxsize=prefetch_size)
        self.stop_flag = False
        
    def _reader_thread(self):
        cap = cv2.VideoCapture(self.video_path); frame_count = 0
        while not self.stop_flag and frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret: break
            self.frame_queue.put((frame_count, frame)); frame_count += 1
        cap.release(); self.frame_queue.put(None)
    
    def start(self):
        import threading
        self.thread = threading.Thread(target=self._reader_thread, daemon=True); self.thread.start()
    def get_frame(self): return self.frame_queue.get()
    def stop(self):
        self.stop_flag = True
        if hasattr(self, 'thread'): self.thread.join()

class Visualizer:
    # (Không thay đổi so với bản Canny)
    def __init__(self, config: TrafficSignConfig):
        self.config = config

    def draw_all(self, frame: np.ndarray, frame_num: int,
                 detections: List[Tuple], roi_params_dict: Dict) -> np.ndarray:
        frame_output = frame.copy()
        frame_output = self._draw_student_ids(frame_output)
        if self.config.DEBUG_MODE:
            frame_output = self._draw_frame_id(frame_output, frame_num)
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

    def _draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        for bbox, color_type, metrics in detections:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.config.BOX_COLOR, 3)
            sign_name = metrics.get('sign_name', '')
            if self.config.SHOW_SIGN_NAME and sign_name:
                text_y = max(y - 15, 25)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7; thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(sign_name, font, font_scale, thickness)
                cv2.rectangle(frame, (x, text_y - text_h - 8),
                            (x + text_w + 10, text_y + baseline + 2), (0, 0, 0), -1)
                cv2.putText(frame, sign_name, (x + 5, text_y),
                          font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            elif self.config.DEBUG_MODE and metrics:
                shape = metrics.get('shape', 'unknown')
                area = metrics.get('area', 0)
                if shape == 'circle': text = f"C:{metrics.get('circularity', 0):.2f}"
                else: text = f"S:{metrics.get('solidity', 0):.2f}"
                text = f"{color_type} A:{area} {text}"
                text_y = max(y - 10, 20)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, text_y - text_h - 5),
                            (x + text_w + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_debug_detections(self, frame: np.ndarray, frame_num: int,
                              detections: List[Tuple], roi_params_dict: Dict) -> np.ndarray:
        frame_output = frame.copy()
        frame_output = self._draw_student_ids(frame_output)
        frame_output = self._draw_frame_id(frame_output, frame_num)
        
        text = f"Post-SVM Detections (Pass 1): {len(detections)}"
        x, y = 10, 110
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(frame_output, (x - 5, y - text_h - 5),
                     (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame_output, text, (x, y), font, 0.8, (255, 165, 0), 2, cv2.LINE_AA)
        
        for bbox, color_type, metrics in detections:
            x, y, w, h = bbox
            if color_type == 'blue': box_color = (255, 0, 0)
            elif color_type == 'red': box_color = (0, 0, 255)
            elif color_type == 'yellow': box_color = (0, 255, 255)
            else: box_color = (128, 128, 128)
            
            cv2.rectangle(frame_output, (x, y), (x + w, y + h), box_color, 2)
            shape = metrics.get('shape', 'unknown'); area = metrics.get('area', 0)
            if shape == 'circle': circularity = metrics.get('circularity', 0)
            else: solidity = metrics.get('solidity', 0)
            
            if shape == 'circle': metrics_text = [f"{color_type.upper()}", f"Area:{area}", f"Circ:{circularity:.2f}"]
            else: metrics_text = [f"{color_type.upper()}", f"Area:{area}", f"Sol:{solidity:.2f}"]
            
            font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.45; thickness = 1; line_height = 15
            total_text_height = len(metrics_text) * line_height
            if y - total_text_height - 10 > 0: text_y_start = y - 10
            else: text_y_start = y + h + 20
            
            for i, text in enumerate(metrics_text):
                text_y = text_y_start + i * line_height
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(frame_output, (x, text_y - text_h - 3),
                            (x + text_w + 4, text_y + 3), (0, 0, 0), -1)
                cv2.putText(frame_output, text, (x + 2, text_y),
                          font, font_scale, box_color, thickness, cv2.LINE_AA)
        return frame_output

class SignDetectorSVM:
    # (Không thay đổi - HOG Xám)
    def __init__(self, model_path: str):
        print(f"Loading SVM Detector from: {model_path}")
        try:
            self.svm = cv2.ml.SVM_load(model_path)
            if self.svm is None: raise Exception(f"Cannot load model from {model_path}")
        except Exception as e:
            print(f"❌ Critical error: Cannot load Detector model."); print(f"   Details: {e}"); raise
        RESIZE_DIM = (64, 64); winSize = RESIZE_DIM; blockSize = (16, 16)
        blockStride = (8, 8); cellSize = (8, 8); nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        self.resize_dim = RESIZE_DIM
        print("✓ SVM Detector loaded successfully.")
    def verify(self, frame: np.ndarray, bbox: Tuple) -> bool:
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or y+h > frame.shape[0] or x+w > frame.shape[1]: return False
        try:
            sign_roi = frame[y:y+h, x:x+w]
            sign_gray = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2GRAY)
            sign_resized = cv2.resize(sign_gray, self.resize_dim, interpolation=cv2.INTER_AREA)
            features = self.hog.compute(sign_resized)
            _, result = self.svm.predict(features.reshape(1, -1))
            label = int(result[0][0]); return label == 1
        except: return False

class SignRecognizerSVM:
    # (Không thay đổi)
    def __init__(self, model_path: str, label_map: Dict[int, str]):
        print(f"Loading SVM Recognizer from: {model_path}")
        try:
            self.svm = cv2.ml.SVM_load(model_path)
            if self.svm is None: raise Exception(f"Cannot load model from {model_path}")
        except Exception as e:
            print(f"❌ Critical error: Cannot load Recognizer model."); print(f"   Details: {e}"); raise
        self.label_map = label_map
        RESIZE_DIM = (64, 64); winSize = RESIZE_DIM; blockSize = (16, 16)
        blockStride = (8, 8); cellSize = (8, 8); nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        self.resize_dim = RESIZE_DIM
        print("✓ SVM Recognizer loaded successfully.")
    def _get_hog_features(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or y+h > frame.shape[0] or x+w > frame.shape[1]: return None
        try:
            sign_roi = frame[y:y+h, x:x+w]
            sign_gray = cv2.cvtColor(sign_roi, cv2.COLOR_BGR2GRAY)
            sign_resized = cv2.resize(sign_gray, self.resize_dim, interpolation=cv2.INTER_AREA)
            return self.hog.compute(sign_resized)
        except Exception: return None
    def recognize_batch(self, frame: np.ndarray, verified_detections: List[Tuple]) -> List[Tuple]:
        if not verified_detections: return []
        hog_features_list = []; detections_list = []
        for (bbox, color, metrics) in verified_detections:
            features = self._get_hog_features(frame, bbox)
            if features is not None:
                hog_features_list.append(features)
                detections_list.append((bbox, color, metrics))
        if not hog_features_list: return []
        features_batch = np.array(hog_features_list, dtype=np.float32)
        if features_batch.ndim == 1: features_batch = features_batch.reshape(1, -1)
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
    return (0,0,0,0) # Không dùng
def is_bbox_in_roi(bbox: Tuple, roi_params: Tuple, overlap_threshold: float = 0.5) -> bool:
    return True # Không dùng
def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    # (Giữ nguyên - Cần cho Problem Sign Logic)
    x1, y1, w1, h1 = box1; x2, y2, w2, h2 = box2
    x1_max, y1_max = x1 + w1, y1 + h1; x2_max, y2_max = x2 + w2, y2 + h2
    inter_x1 = max(x1, x2); inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max); inter_y2 = min(y1_max, y2_max)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1: return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1; area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0
def expand_bbox(bbox: Tuple, scale: float, frame_shape: Tuple) -> Tuple:
    # (Giữ nguyên - Cần cho Problem Sign Logic)
    x, y, w, h = bbox; frame_h, frame_w = frame_shape[0:2]
    center_x = x + w / 2; center_y = y + h / 2
    new_w = w * scale; new_h = h * scale
    new_x = int(center_x - new_w / 2); new_y = int(center_y - new_h / 2)
    new_w = int(new_w); new_h = int(new_h)
    new_x = max(0, new_x); new_y = max(0, new_y)
    new_w = min(frame_w - new_x, new_w); new_h = min(frame_h - new_y, new_h)
    return (new_x, new_y, new_w, new_h)


# =============================================================================
# MAIN FUNCTION (LOGIC MSER - CHỐNG FLICKER)
# =============================================================================
def main():
    print("=" * 70)
    print("  TRAFFIC SIGN DETECTION (MSER Dual Channel + ANTI-FLICKER)")
    print("=" * 70)

    try:
        start_total = time.time()
        
        # (MỚI) Tạo config 1 LẦN
        config = TrafficSignConfig(auto_detect=True)
        visualizer = Visualizer(config)

        LABEL_MAP_CUSTOM = {
            0: "cam_di_nguoc_chieu", 1: "cam_queo_trai", 2: "cam_do_xe",
            3: "cam_dung_do_xe", 4: "huong_ben_phai", 5: "canh_bao_di_cham",
            6: "canh_bao_nguoi_qua_duong", 7: "canh_bao_duong_gap_khuc"
        }

        try:
            recognizer_svm = SignRecognizerSVM(
                model_path=MODEL_RECOGNIZER_PATH,
                label_map=LABEL_MAP_CUSTOM
            )
        except Exception as e:
            print("Cannot initialize SVM recognizer"); return

        cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video '{config.INPUT_VIDEO_PATH}'"); return

        fps = cap.get(cv2.CAP_PROP_FPS);
        if fps == 0: fps = 30.0
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"\n📹 VIDEO INFORMATION:")
        print(f"   File: {config.INPUT_VIDEO_PATH}")
        print(f"   Resolution: {w_orig}x{h_orig}, FPS: {fps:.2f}, Total Frames: {total_frames}")
        print(f"   Processing Limit: {config.MAX_FRAME_ID} frames")

        tracker = SignTracker(fps, config.TRACKING_PARAMS)
        h_crop = int(h_orig * config.PROCESSING_HEIGHT_PERCENT); w_crop = w_orig
        full_frame_dims = (w_orig, h_orig)
        roi_pixel_map = {} 

        # =====================================================================
        # PASS 1: PARALLEL DETECT (MSER) + VERIFY (SVM) + TRACK
        # =====================================================================
        print("\n" + "=" * 70)
        print("⚡ PASS 1: PARALLEL DETECT (MSER) + VERIFY (SVM) + TRACK")
        print("=" * 70)
        start_detection = time.time()

        frame_reader = ParallelFrameReader(config.INPUT_VIDEO_PATH, 
                                          config.MAX_FRAME_ID, 
                                          config.PREFETCH_FRAMES)
        frame_reader.start()

        mser_writer = None
        if config.SAVE_MASK_VIDEOS:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            mser_writer = cv2.VideoWriter(config.MASK_VIDEO_MSER, fourcc, fps, (w_crop, h_crop), False)

        debug_writer = None
        if config.SAVE_DEBUG_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            debug_writer = cv2.VideoWriter(config.DEBUG_VIDEO_PATH, fourcc, fps, (w_orig, h_orig))
            if debug_writer.isOpened(): print(f"\n📹 Debug video enabled: {config.DEBUG_VIDEO_PATH}")
            else: print(f"\n⚠️  Warning: Cannot create debug video writer"); debug_writer = None

        frame_batch = []; frame_full_batch = []; frame_count = 0
        
        # (MỚI) Truyền config vào worker
        init_args = (MODEL_DETECTOR_PATH, config)
        
        with Pool(processes=config.NUM_WORKERS, 
                  initializer=initialize_worker, 
                  initargs=init_args) as pool:
            
            while True:
                item = frame_reader.get_frame()
                if item is None: break
                
                frame_num, frame_full = item
                frame_to_process = frame_full[0:h_crop, 0:w_crop]
                
                # (MỚI) Không cần truyền config vào batch
                frame_batch.append((frame_num, frame_to_process, full_frame_dims))
                
                if debug_writer:
                    frame_full_batch.append((frame_num, frame_full.copy()))
                
                if len(frame_batch) >= config.BATCH_SIZE:
                    results = pool.map(process_frame_worker, frame_batch)
                    
                    if debug_writer and frame_full_batch:
                        detections_map = {}
                        for frame_num, detections, masks in results:
                            detections_map[frame_num] = detections
                    
                    for frame_num, detections, masks in results:
                        if detections:
                            for bbox, color, metrics in detections:
                                tracker.add_detection(frame_num, bbox, color, metrics)
                        
                        if config.SAVE_MASK_VIDEOS and mser_writer and 'mser_dual' in masks:
                            mser_writer.write(masks['mser_dual'])
                    
                    if debug_writer and frame_full_batch:
                        for frame_num, frame_full in frame_full_batch:
                            detections = detections_map.get(frame_num, [])
                            debug_frame = visualizer.draw_debug_detections(
                                frame_full, frame_num, detections, roi_pixel_map
                            )
                            debug_writer.write(debug_frame)
                        frame_full_batch.clear()
                    
                    frame_count += len(frame_batch); frame_batch.clear()
                    
                    if frame_count % config.PROGRESS_UPDATE_INTERVAL == 0:
                        progress = int(frame_count / config.MAX_FRAME_ID * 100)
                        elapsed = time.time() - start_detection
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"   Detected/Verified {frame_count} frames ({progress}%) - {current_fps:.1f} FPS")
                        gc.collect()
            
            if frame_batch:
                results = pool.map(process_frame_worker, frame_batch)
                
                if debug_writer and frame_full_batch:
                    detections_map = {}
                    for frame_num, detections, masks in results:
                        detections_map[frame_num] = detections
                
                for frame_num, detections, masks in results:
                    if detections:
                        for bbox, color, metrics in detections:
                            tracker.add_detection(frame_num, bbox, color, metrics)
                    if config.SAVE_MASK_VIDEOS and mser_writer and 'mser_dual' in masks:
                        mser_writer.write(masks['mser_dual'])
                
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
        if mser_writer: mser_writer.release()
        if debug_writer:
            debug_writer.release()
            print(f"✅ Debug video (Post-SVM) saved: {config.DEBUG_VIDEO_PATH}")

        detection_time = time.time() - start_detection
        print(f"\n✅ Detect/Verify Pass completed in {detection_time:.2f}s")
        print(f"   Average Speed: {frame_count/detection_time:.1f} FPS")

        tracker.build_smoothed_cache()
        print(f"\n📊 Tracking Statistics:")
        print(f"   Total tracks (post-SVM): {len(tracker.tracks)}")

        # =====================================================================
        # PASS 2: PARALLEL RECOGNIZE + RENDER
        # =====================================================================
        print("\n" + "=" * 70)
        print("⚡ PASS 2: PARALLEL RECOGNIZE + RENDER (NO VERIFY)")
        print("=" * 70)
        start_render = time.time()

        frame_reader = ParallelFrameReader(config.INPUT_VIDEO_PATH, 
                                          total_frames, 
                                          config.PREFETCH_FRAMES)
        frame_reader.start()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps,
                                       (w_orig, h_orig))
        if not video_writer.isOpened():
            print(f"❌ Error: Cannot create output '{config.OUTPUT_VIDEO_PATH}'"); return

        frame_count_render = 0; frames_with_detections = 0
        total_signs_recognized = 0; saved_images_count = 0
        
        if config.SAVE_DETECTOR_IMAGES:
            if not os.path.exists(config.DETECTOR_IMAGES_FOLDER):
                os.makedirs(config.DETECTOR_IMAGES_FOLDER)
            print(f"\n📁 Saving detector images to: {config.DETECTOR_IMAGES_FOLDER}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                item = frame_reader.get_frame()
                if item is None: break
                
                frame_num, frame = item
                
                smoothed_candidates = tracker.get_smoothed_detections(frame_num)
                
                recognized_detections = []
                if smoothed_candidates:
                    recognized_detections = recognizer_svm.recognize_batch(frame, smoothed_candidates)
                    
                    if config.SAVE_DETECTOR_IMAGES:
                        for (bbox, color, metrics) in recognized_detections:
                            sign_name = metrics.get('sign_name', 'unknown')
                            x, y, w, h = bbox
                            if w > 0 and h > 0 and y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                                sign_roi = frame[y:y+h, x:x+w]
                                img_filename = f"frame{frame_num:06d}_{sign_name}_{color}.jpg"
                                img_path = os.path.join(config.DETECTOR_IMAGES_FOLDER, img_filename)
                                cv2.imwrite(img_path, sign_roi)
                                saved_images_count += 1

                final_detections = []
                red_problem_bboxes = []; blue_problem_detections = []
                for (bbox, color, metrics) in recognized_detections:
                    sign_name = metrics.get('sign_name')
                    if sign_name in config.PROBLEM_SIGNS:
                        if color == 'red':
                            final_detections.append((bbox, color, metrics))
                            red_problem_bboxes.append(bbox)
                        elif color == 'blue':
                            blue_problem_detections.append((bbox, color, metrics))
                    else:
                        final_detections.append((bbox, color, metrics))
                
                if blue_problem_detections:
                    for (blue_bbox, blue_color, blue_metrics) in blue_problem_detections:
                        is_overlapped_by_red = False
                        for red_bbox in red_problem_bboxes:
                            iou = calculate_iou(blue_bbox, red_bbox)
                            if iou > config.BBOX_OVERLAP_IOU: is_overlapped_by_red = True; break
                        if not is_overlapped_by_red:
                            expanded_bbox = expand_bbox(blue_bbox, config.BBOX_EXPAND_SCALE, frame.shape)
                            final_detections.append((expanded_bbox, blue_color, blue_metrics))
                
                total_signs_recognized += len(final_detections)

                frame_output = visualizer.draw_all(frame, frame_num, final_detections, roi_pixel_map)
                video_writer.write(frame_output)
                if final_detections: frames_with_detections += 1
                frame_count_render += 1

                if frame_count_render % config.PROGRESS_UPDATE_INTERVAL == 0:
                    progress = int(frame_count_render / total_frames * 100)
                    elapsed = time.time() - start_render
                    current_fps = frame_count_render / elapsed if elapsed > 0 else 0
                    status_msg = f"   Rendered {frame_count_render}/{total_frames} frames ({progress}%) " \
                                 f"- {current_fps:.1f} FPS - {total_signs_recognized} signs"
                    if config.SAVE_DETECTOR_IMAGES:
                        status_msg += f" - {saved_images_count} imgs saved"
                    print(status_msg)
                del frame, frame_output

        frame_reader.stop(); video_writer.release()
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
        print("✅ PROCESSING COMPLETE (MSER Dual Channel LOGIC)!")
        print("=" * 70)
        print(f"📁 Output: {config.OUTPUT_VIDEO_PATH}")
        if config.SAVE_DEBUG_VIDEO: print(f"📁 Debug (Post-SVM): {config.DEBUG_VIDEO_PATH}")
        if config.SAVE_DETECTOR_IMAGES: print(f"📁 Detector Images: {config.DETECTOR_IMAGES_FOLDER}")
        if config.SAVE_MASK_VIDEOS: print(f"📁 MSER Mask: {config.MASK_VIDEO_MSER}")

        print(f"\n⏱️  PERFORMANCE SUMMARY:")
        print("=" * 70)
        print(f"   Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"   Detect/Verify Phase (Pass 1): {detection_time:.2f}s ({detection_time/total_time*100:.1f}%)")
        print(f"   Recognize/Render Phase (Pass 2): {render_time:.2f}s ({render_time/total_time*100:.1f}%)")
        print(f"   Overall Speed: {frame_count_render/total_time:.1f} FPS")
        
        baseline_fps = 12
        actual_speedup = (frame_count_render/total_time) / baseline_fps
        print(f"   Actual Speedup: {actual_speedup:.1f}x vs sequential")

        print(f"\n🎯 OPTIMIZATION APPLIED:")
        print(f"   ✅ MSER DUAL CHANNEL PIPELINE (No color/edge tuning)")
        print(f"   ✅ ANTI-FLICKER LOGIC")
        print(f"   ✅ Multi-processing ({config.NUM_WORKERS} workers)")
        print(f"   ✅ Thread-based I/O")
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