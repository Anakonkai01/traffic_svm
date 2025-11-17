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

"""
TRAFFIC SIGN DETECTION - FINAL COMPLETE VERSION (MSER Dual + NMS + BatchSVM)
============================================================================
1. MSER Dual Channel: Chạy riêng trên Cr và Cb để bắt mọi sắc độ màu.
2. NMS: Loại bỏ khung hình trùng lặp.
3. Batch SVM: Tối ưu hóa tốc độ kiểm tra.
4. Logic: Anti-Flicker, Problem Signs (Cấm đỗ xe) được giữ nguyên.
"""

DRIVE_PATH = "../"
MODEL_DETECTOR_PATH = DRIVE_PATH + "models/svm_sign_detector_k20.xml"
MODEL_RECOGNIZER_PATH = DRIVE_PATH + "models/svm_sign_recognizer_k20.xml"

# =============================================================================
# 1. AUTO-DETECTION SYSTEM (ĐẦY ĐỦ)
# =============================================================================
class HardwareDetector:
    def __init__(self):
        self.cpu_cores = cpu_count()
        self.ram_gb = self._get_ram_gb()
        self.has_gpu = self._check_gpu()
        self.storage_type = self._detect_storage_type()
        self.cpu_model = self._get_cpu_model()
        self.os_info = self._get_os_info()
        
    def _get_ram_gb(self) -> float:
        try: return psutil.virtual_memory().total / (1024 ** 3)
        except: return 8.0
    
    def _check_gpu(self) -> bool:
        try: return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except: return False
        
    def _detect_storage_type(self) -> str:
        try:
            if platform.system() == 'Windows':
                import subprocess
                output = subprocess.check_output('powershell "Get-PhysicalDisk | Select MediaType"', shell=True, text=True)
                if 'SSD' in output: return 'SSD'
            return 'SSD' 
        except: return 'SSD'

    def _get_cpu_model(self) -> str:
        try:
            if platform.system() == 'Windows': return platform.processor()
            elif platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line: return line.split(':')[1].strip()
            return platform.processor()
        except: return "Unknown CPU"

    def _get_os_info(self) -> str:
        return f"{platform.system()} {platform.release()}"
    
    def get_optimal_config(self) -> Dict:
        config = {}
        if self.cpu_cores <= 4:
            config['num_workers'] = max(1, self.cpu_cores - 1)
            config['batch_size'] = 8
            config['prefetch_frames'] = 32
        elif self.cpu_cores <= 8:
            config['num_workers'] = max(1, self.cpu_cores - 1)
            config['batch_size'] = 16
            config['prefetch_frames'] = 64
        else:
            config['num_workers'] = max(1, self.cpu_cores - 2)
            config['batch_size'] = 24
            config['prefetch_frames'] = 96
            
        if self.ram_gb < 8:
            config['batch_size'] = max(4, config['batch_size'] // 2)
        
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
        print(f"\n📊 Optimal Config: Workers={config['num_workers']}, Batch={config['batch_size']}")
        print("=" * 70)

    def print_optimized_settings(self, config: Dict):
        print("\n⚙️  AUTO-CONFIGURED SETTINGS:")
        print("=" * 70)
        print(f"   Worker Processes: {config['num_workers']}")
        print(f"   Batch Size: {config['batch_size']}")
        print(f"   Prefetch Buffer: {config['prefetch_frames']}")
        print(f"   GPU Acceleration: {'✅ Enabled' if config['use_gpu'] else '❌ Disabled'}")
        print("=" * 70)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class TrafficSignConfig:
    def __init__(self, auto_detect: bool = True):
        self.INPUT_VIDEO_PATH = DRIVE_PATH + 'videos/video2.mp4'
        self.OUTPUT_VIDEO_PATH = DRIVE_PATH + 'videos/video_output_final.mp4'
        self.STUDENT_IDS = "523H0164_523H0177_523H0145"

        self.SAVE_MASK_VIDEOS = True
        self.MASK_VIDEO_MSER = DRIVE_PATH + 'videos/mask_mser_dual.mp4'
        self.SAVE_DEBUG_VIDEO = True
        self.DEBUG_VIDEO_PATH = DRIVE_PATH + 'videos/debug_final.mp4'
        self.SAVE_DETECTOR_IMAGES = True
        self.DETECTOR_IMAGES_FOLDER = DRIVE_PATH + 'check_data_final'

        self.MAX_FRAME_ID = 10000
        self.PROGRESS_UPDATE_INTERVAL = 100

        if auto_detect:
            self.hardware = HardwareDetector()
            self.hardware.print_hardware_report()
            opt = self.hardware.get_optimal_config()
            self.NUM_WORKERS = opt['num_workers']
            self.BATCH_SIZE = opt['batch_size']
            self.PREFETCH_FRAMES = opt['prefetch_frames']
            self.USE_GPU = opt['use_gpu']
            self.hardware.print_optimized_settings(opt)
        else:
            self.NUM_WORKERS = 4; self.BATCH_SIZE = 16; self.PREFETCH_FRAMES = 64; self.USE_GPU = False
        self.USE_THREADING = True

        self.DEBUG_MODE = True
        self.BOX_COLOR = (0, 255, 0)
        self.SHOW_SIGN_NAME = True
        self.PROCESSING_HEIGHT_PERCENT = 1.0

        # === MSER PARAMETERS ===
        self.MSER_DELTA = 4      
        self.MSER_MIN_AREA = 50      
        self.MSER_MAX_AREA = 14400
        self.MSER_MAX_VARIATION = 0.3
        self.MSER_MIN_DIVERSITY = 0.2
        
        # === COLOR GUESS (YCrCb-MEAN) ===
        self.COLOR_GUESS_CR_MIN = 125 
        self.COLOR_GUESS_CB_MIN = 135 
        self.COLOR_GUESS_CR_MAX_FOR_YELLOW = 130
        self.COLOR_GUESS_CB_MAX_FOR_YELLOW = 120

        # === SIZE FILTERING ===
        self.MIN_SIGN_WIDTH = 20
        self.MIN_SIGN_HEIGHT = 20
        self.MAX_ASPECT_RATIO = 2.5 
        self.MIN_ASPECT_RATIO = 0.4

        # === SHAPE PARAMETERS ===
        self.SHAPE_PARAMS = {
            'circle': {
                'min_area': 200, 'max_area': 15000,
                'trust_threshold': 1000,
                'small_circularity': 0.72, 
                'large_circularity': 0.8
            },
            'triangle': {
                'min_area': 200, 'max_area': 50000,
                'trust_threshold': 1500,
                'min_solidity': 0.65, 
                'epsilon_factor': 0.035,
                'max_vertices': 7
            }
        }

        # === TRACKING & NMS ===
        self.TRACKING_PARAMS = {
            'max_gap_sec': 1.0,
            'iou_threshold': 0.2, 
            'smoothing_window': 15
        }
        self.NMS_IOU_THRESHOLD = 0.3 
        
        self.PROBLEM_SIGNS = {"cam_do_xe", "cam_dung_do_xe"}
        self.BBOX_EXPAND_SCALE = 1.3 
        self.BBOX_OVERLAP_IOU = 0.3

# =============================================================================
# 3. UTILITY FUNCTIONS
# =============================================================================
def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    x1, y1, w1, h1 = box1; x2, y2, w2, h2 = box2
    inter_x1 = max(x1, x2); inter_y1 = max(y1, y2)
    inter_x2 = min(x1+w1, x2+w2); inter_y2 = min(y1+h1, y2+h2)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1: return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = w1 * h1; area2 = w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def non_max_suppression(detections: List[Tuple], iou_threshold: float) -> List[Tuple]:
    """Loại bỏ các BBox trùng lặp dựa trên IoU."""
    if not detections: return []
    # Ưu tiên BBox có diện tích lớn hơn
    detections = sorted(detections, key=lambda x: x[0][2]*x[0][3], reverse=True)
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        # Loại bỏ các box trùng với current quá nhiều
        detections = [d for d in detections if calculate_iou(current[0], d[0]) < iou_threshold]
    return keep

def expand_bbox(bbox: Tuple, scale: float, frame_shape: Tuple) -> Tuple:
    x, y, w, h = bbox; frame_h, frame_w = frame_shape[:2]
    cx, cy = x + w/2, y + h/2
    nw, nh = w * scale, h * scale
    nx, ny = int(cx - nw/2), int(cy - nh/2)
    nw, nh = int(nw), int(nh)
    nx, ny = max(0, nx), max(0, ny)
    nw, nh = min(frame_w - nx, nw), min(frame_h - ny, nh)
    return (nx, ny, nw, nh)

def convert_roi_to_pixels(roi_percent: Tuple, w_full: int, h_full: int) -> Tuple:
    x_start_pct, y_start_pct, x_end_pct, y_end_pct = roi_percent
    return (int(w_full * x_start_pct), int(h_full * y_start_pct),
            int(w_full * x_end_pct), int(h_full * y_end_pct))

# =============================================================================
# 4. CLASS: TRACKING
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

    def add_detection(self, frame_num: int, bbox: Tuple, color: str, metrics: Dict):
        self._cache_built = False
        best_match_id = None; best_iou = 0
        for track_id, track_data in self.tracks.items():
            if not track_data: continue
            last = track_data[-1]
            if last['color'] == color and (frame_num - last['frame'] <= self.max_gap_frames):
                iou = calculate_iou(bbox, last['bbox'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou; best_match_id = track_id
        
        if best_match_id is not None:
            self.tracks[best_match_id].append({'frame': frame_num, 'bbox': bbox, 'color': color, 'metrics': metrics})
        else:
            self.tracks[self.next_track_id] = [{'frame': frame_num, 'bbox': bbox, 'color': color, 'metrics': metrics}]
            self.next_track_id += 1

    def interpolate_and_smooth(self):
        if self._cache_built: return
        print("   Building smoothed detection cache...")
        self._smoothed_cache.clear()
        for track_id, track_data in self.tracks.items():
            if len(track_data) < 2: continue
            
            # Interpolate
            filled_track = []
            for i in range(len(track_data)-1):
                curr, next_det = track_data[i], track_data[i+1]
                filled_track.append(curr)
                gap = next_det['frame'] - curr['frame']
                if 1 < gap <= self.max_gap_frames:
                    for j in range(1, gap):
                        alpha = j / gap
                        bx = tuple(int(curr['bbox'][k] + (next_det['bbox'][k]-curr['bbox'][k])*alpha) for k in range(4))
                        filled_track.append({'frame': curr['frame']+j, 'bbox': bx, 'color': curr['color'], 'metrics': {}, 'interpolated': True})
            filled_track.append(track_data[-1])
            
            # Smooth
            for i in range(len(filled_track)):
                start = max(0, i - self.smoothing_window//2)
                end = min(len(filled_track), i + self.smoothing_window//2 + 1)
                segment = filled_track[start:end]
                avg_bbox = tuple(int(np.mean([d['bbox'][k] for d in segment])) for k in range(4))
                
                frame_num = filled_track[i]['frame']
                if frame_num not in self._smoothed_cache: self._smoothed_cache[frame_num] = []
                self._smoothed_cache[frame_num].append((avg_bbox, filled_track[i]['color'], filled_track[i].get('metrics', {})))
        self._cache_built = True

    def get_smoothed_detections(self, frame_num: int) -> List[Tuple]:
        if not self._cache_built: return []
        return self._smoothed_cache.get(frame_num, [])

# =============================================================================
# 5. CLASS: DETECTOR (MSER DUAL CHANNEL)
# =============================================================================
class TrafficSignDetector:
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        
    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tách kênh YCrCb"""
        frame_blur = cv2.medianBlur(frame, 5)
        frame_ycrcb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(frame_ycrcb)
        return Y, Cr, Cb, frame_ycrcb
    
    def _validate_size(self, bbox: Tuple) -> bool:
        """Lọc kích thước sớm"""
        w, h = bbox[2], bbox[3]
        if w < self.config.MIN_SIGN_WIDTH or h < self.config.MIN_SIGN_HEIGHT: return False
        ratio = w / h
        if ratio < self.config.MIN_ASPECT_RATIO or ratio > self.config.MAX_ASPECT_RATIO: return False
        return True

    def _get_dominant_color(self, roi_ycrcb: np.ndarray) -> str:
        try:
            mean_vals = cv2.mean(roi_ycrcb)
            cr, cb = mean_vals[1], mean_vals[2]
            if cr > self.config.COLOR_GUESS_CR_MIN: return 'red'
            if cb > self.config.COLOR_GUESS_CB_MIN: return 'blue'
            if cr < self.config.COLOR_GUESS_CR_MAX_FOR_YELLOW and cb < self.config.COLOR_GUESS_CB_MAX_FOR_YELLOW: return 'yellow'
            return 'unknown'
        except: return 'unknown'

    def _detect_shape(self, contour: np.ndarray, area: float) -> Tuple[str, Dict]:
        hull = cv2.convexHull(contour)
        p_hull = cv2.arcLength(hull, True)
        if p_hull <= 0: return None, {}
        
        # Check Circle
        circularity = 4 * np.pi * (cv2.contourArea(hull) / (p_hull ** 2))
        if circularity > self.config.SHAPE_PARAMS['circle']['small_circularity']:
             return 'circle', {'area': int(area), 'circularity': circularity, 'shape': 'circle'}
        
        # Check Triangle
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        if solidity > self.config.SHAPE_PARAMS['triangle']['min_solidity']:
             approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
             if len(approx) <= 7:
                 return 'triangle', {'area': int(area), 'solidity': solidity, 'shape': 'triangle'}
        return None, {}

    def process_frame(self, frame: np.ndarray, full_frame_dims: Tuple, mser_cr: cv2.MSER, mser_cb: cv2.MSER) -> Tuple[List[Tuple], Dict]:
        """Pipeline MSER Dual Channel"""
        all_detections = []
        masks_dict = {}

        # 1. Tách các kênh
        Y, Cr, Cb, frame_ycrcb = self._preprocess_frame(frame)
        
        # 2. Dual Channel MSER
        regions_cr, bboxes_cr = mser_cr.detectRegions(Cr)
        regions_cb, bboxes_cb = mser_cb.detectRegions(Cb)
        
        # 3. Kết hợp kết quả
        all_regions = []
        all_bboxes = []
        if regions_cr is not None:
            all_regions.extend(regions_cr)
            all_bboxes.extend(list(bboxes_cr))
        if regions_cb is not None:
            all_regions.extend(regions_cb)
            all_bboxes.extend(list(bboxes_cb))
        
        # Debug Mask
        mser_mask = np.zeros_like(Y)
        if all_regions: cv2.drawContours(mser_mask, all_regions, -1, (255), 1)
        masks_dict['mser_dual'] = mser_mask

        # 4. Filter & Process
        for contour, bbox in zip(all_regions, all_bboxes):
            if not self._validate_size(bbox): continue
            area = cv2.contourArea(contour)
            
            shape_type, metrics = self._detect_shape(contour, area)
            if shape_type:
                x, y, w, h = bbox
                roi = frame_ycrcb[y:y+h, x:x+w]
                color = self._get_dominant_color(roi)
                all_detections.append((tuple(bbox), color, metrics))
        
        return all_detections, masks_dict

# =============================================================================
# 6. SVM MODELS (Với Batch Verify)
# =============================================================================
class SignDetectorSVM:
    def __init__(self, path):
        self.svm = cv2.ml.SVM_load(path)
        self.hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
        self.resize_dim = (64, 64)

    def verify(self, frame, bbox):
        x,y,w,h = bbox
        if w<=0 or h<=0 or y+h>frame.shape[0] or x+w>frame.shape[1]: return False
        try:
            roi = cv2.resize(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY), self.resize_dim)
            feat = self.hog.compute(roi)
            return int(self.svm.predict(feat.reshape(1,-1))[1][0][0]) == 1
        except: return False

    def verify_batch(self, frame, detections):
        """Batch Verification"""
        if not detections: return []
        hogs = []; valid_indices = []
        
        for i, (bbox, _, _) in enumerate(detections):
            x, y, w, h = bbox
            if w<=0 or h<=0 or y+h>frame.shape[0] or x+w>frame.shape[1]: continue
            try:
                roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi, self.resize_dim)
                feat = self.hog.compute(roi_resized)
                hogs.append(feat)
                valid_indices.append(i)
            except: continue
            
        if not hogs: return []
        hogs_batch = np.array(hogs, dtype=np.float32)
        if hogs_batch.ndim == 3: hogs_batch = hogs_batch.reshape(hogs_batch.shape[0], -1)
            
        _, results = self.svm.predict(hogs_batch)
        
        final_detections = []
        for idx_in_batch, original_idx in enumerate(valid_indices):
            label = int(results[idx_in_batch][0])
            if label == 1:
                final_detections.append(detections[original_idx])
        return final_detections

class SignRecognizerSVM:
    def __init__(self, path, labels):
        self.svm = cv2.ml.SVM_load(path)
        self.hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
        self.labels = labels
    def recognize_batch(self, frame, dets):
        if not dets: return []
        feats = []; valid_dets = []
        for d in dets:
            x,y,w,h = d[0]
            if w>0 and h>0 and y+h<=frame.shape[0] and x+w<=frame.shape[1]:
                roi = cv2.resize(cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY), (64,64))
                feats.append(self.hog.compute(roi))
                valid_dets.append(d)
        if not feats: return []
        feats = np.array(feats, dtype=np.float32).reshape(len(feats), -1)
        results = self.svm.predict(feats)[1]
        final = []
        for i, (bbox, color, metrics) in enumerate(valid_dets):
            lid = int(results[i][0])
            metrics['sign_name'] = self.labels.get(lid, f"Unknown_{lid}")
            final.append((bbox, color, metrics))
        return final

# =============================================================================
# 7. WORKER & PROCESSOR
# =============================================================================
worker_models = {}

def initialize_worker(detector_model_path, config: TrafficSignConfig):
    global worker_models
    if not worker_models:
        print(f"Worker {os.getpid()} init...")
        worker_models['config'] = config
        worker_models['detector_svm'] = SignDetectorSVM(detector_model_path)
        # Dual Channel MSER instances
        worker_models['mser_cr'] = cv2.MSER_create(
            delta=config.MSER_DELTA, min_area=config.MSER_MIN_AREA, max_area=config.MSER_MAX_AREA,
            max_variation=config.MSER_MAX_VARIATION, min_diversity=config.MSER_MIN_DIVERSITY
        )
        worker_models['mser_cb'] = cv2.MSER_create(
            delta=config.MSER_DELTA, min_area=config.MSER_MIN_AREA, max_area=config.MSER_MAX_AREA,
            max_variation=config.MSER_MAX_VARIATION, min_diversity=config.MSER_MIN_DIVERSITY
        )
        worker_models['detector'] = TrafficSignDetector(config)

def process_frame_worker(args):
    frame_num, frame, full_dims = args
    global worker_models
    
    detector = worker_models['detector']
    mser_cr = worker_models['mser_cr']
    mser_cb = worker_models['mser_cb']
    svm = worker_models['detector_svm']
    cfg = worker_models['config']

    # 1. Detect (MSER)
    raw_dets, masks = detector.process_frame(frame, full_dims, mser_cr, mser_cb)
    
    # 2. NMS (Loại trùng lặp)
    nms_dets = non_max_suppression(raw_dets, cfg.NMS_IOU_THRESHOLD)
    
    # 3. Batch Verify (SVM)
    final_dets = svm.verify_batch(frame, nms_dets)
            
    return frame_num, final_dets, masks

# =============================================================================
# 8. HELPER CLASSES
# =============================================================================
class ParallelFrameReader:
    def __init__(self, path, max_f, prefetch):
        self.cap = cv2.VideoCapture(path)
        self.max_f = max_f
        self.q = queue.Queue(maxsize=prefetch)
        self.stop_f = False
        self.t = None
    def start(self):
        import threading
        self.t = threading.Thread(target=self.run, daemon=True); self.t.start()
    def run(self):
        cnt = 0
        while not self.stop_f and cnt < self.max_f:
            ret, fr = self.cap.read()
            if not ret: break
            self.q.put((cnt, fr)); cnt+=1
        self.q.put(None); self.cap.release()
    def get(self): return self.q.get()
    def stop(self): self.stop_f = True; self.t.join() if self.t else None

# =============================================================================
# CLASS 8: VISUALIZER (ĐÃ BỔ SUNG ĐẦY ĐỦ)
# =============================================================================
class Visualizer:
    def __init__(self, config: TrafficSignConfig):
        self.config = config

    def draw(self, frame: np.ndarray, frame_num: int, detections: List[Tuple]) -> np.ndarray:
        """Vẽ kết quả cuối cùng (cho video output)"""
        frame_output = frame.copy()
        
        # Vẽ Student IDs
        text = self.config.STUDENT_IDS
        cv2.rectangle(frame_output, (5, 5), (450, 40), (0, 0, 0), -1)
        cv2.putText(frame_output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Vẽ Frame ID
        cv2.putText(frame_output, f"Frame: {frame_num}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Vẽ các biển báo đã nhận diện
        for bbox, color, metrics in detections:
            x, y, w, h = bbox
            
            # Màu khung dựa trên loại biển
            if color == 'red': box_color = (0, 0, 255)
            elif color == 'blue': box_color = (255, 0, 0)
            else: box_color = (0, 255, 255) # Yellow

            cv2.rectangle(frame_output, (x, y), (x + w, y + h), box_color, 3)

            # Vẽ tên biển báo
            sign_name = metrics.get('sign_name', '')
            if self.config.SHOW_SIGN_NAME and sign_name:
                text_y = max(y - 15, 25)
                (text_w, text_h), _ = cv2.getTextSize(sign_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame_output, (x, text_y - text_h - 8), (x + text_w + 10, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame_output, sign_name, (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame_output

    def draw_debug_detections(self, frame: np.ndarray, frame_num: int,
                              detections: List[Tuple], roi_params_dict: Dict) -> np.ndarray:
        """
        Vẽ thông tin Debug (cho Giai đoạn 1: Detect & Verify).
        Hiển thị các ứng cử viên đã qua SVM nhưng chưa qua Recognizer.
        """
        frame_output = frame.copy()
        
        # Vẽ Frame info
        cv2.putText(frame_output, f"Post-SVM Candidates: {len(detections)}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        for bbox, color_type, metrics in detections:
            x, y, w, h = bbox
            
            # Màu dựa trên 'Color Guess'
            if color_type == 'blue': box_color = (255, 0, 0)
            elif color_type == 'red': box_color = (0, 0, 255)
            elif color_type == 'yellow': box_color = (0, 255, 255)
            else: box_color = (128, 128, 128)
            
            cv2.rectangle(frame_output, (x, y), (x + w, y + h), box_color, 2)
            
            # Hiển thị thông số kỹ thuật (Area, Shape)
            shape = metrics.get('shape', 'unk')
            area = metrics.get('area', 0)
            
            info_text = f"{color_type} {shape}"
            metric_text = f"A:{area}"
            
            cv2.putText(frame_output, info_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            cv2.putText(frame_output, metric_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        return frame_output
# =============================================================================
# 9. MAIN
# =============================================================================
def main():
    print("🚀 TRAFFIC SIGN DETECTION - FINAL (MSER Dual + NMS + BatchSVM)")
    
    config = TrafficSignConfig(auto_detect=True)
    tracker = SignTracker(30.0, config.TRACKING_PARAMS)
    visualizer = Visualizer(config)
    
    LABELS = {0: "cam_di_nguoc_chieu", 1: "cam_queo_trai", 2: "cam_do_xe", 3: "cam_dung_do_xe", 
              4: "huong_ben_phai", 5: "canh_bao_di_cham", 6: "canh_bao_nguoi_qua_duong", 7: "canh_bao_duong_gap_khuc"}
    
    try:
        recognizer = SignRecognizerSVM(MODEL_RECOGNIZER_PATH, LABELS)
    except: return

    reader = ParallelFrameReader(config.INPUT_VIDEO_PATH, config.MAX_FRAME_ID, config.PREFETCH_FRAMES)
    reader.start()
    
    # Writers
    cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
    w, h = int(cap.get(3)), int(cap.get(4))
    video_writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w,h))
    mser_writer = cv2.VideoWriter(config.MASK_VIDEO_MSER, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w,h), False) if config.SAVE_MASK_VIDEOS else None
    debug_writer = cv2.VideoWriter(config.DEBUG_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w,h)) if config.SAVE_DEBUG_VIDEO else None

    # --- PASS 1: DETECT ---
    print("\n⚡ PASS 1: DETECT (MSER Dual) -> NMS -> Batch Verify (SVM) -> TRACK")
    batch = []
    roi_map = {} # Dummy for compatibility
    
    with Pool(config.NUM_WORKERS, initialize_worker, (MODEL_DETECTOR_PATH, config)) as pool:
        while True:
            item = reader.get()
            if item is None: break
            
            fnum, frame = item
            batch.append((fnum, frame, (w,h)))
            
            if debug_writer: 
                # Note: Đây là frame chưa vẽ, nhưng logic debug thường vẽ đè lên
                # Chúng ta sẽ vẽ debug sau khi có kết quả, hoặc ghi frame gốc
                pass 

            if len(batch) >= config.BATCH_SIZE:
                results = pool.map(process_frame_worker, batch)
                for fn, dets, masks in results:
                    # Ghi MSER Mask
                    if mser_writer: mser_writer.write(masks.get('mser_dual'))
                    
                    # Thêm vào Tracker
                    for d in dets: tracker.add_detection(fn, d[0], d[1], d[2])
                    
                    # Ghi Debug Video (Post-SVM)
                    if debug_writer:
                        # Tìm frame tương ứng trong batch (hơi chậm nếu tìm linear, nhưng OK cho debug)
                        orig_frame = next((b[1] for b in batch if b[0] == fn), None)
                        if orig_frame is not None:
                            debug_frame = visualizer.draw_debug_detections(orig_frame, fn, dets, roi_map)
                            debug_writer.write(debug_frame)

                batch = []
                print(f"Processed frame {fnum}", end='\r')
        
        # Process remaining
        if batch:
            results = pool.map(process_frame_worker, batch)
            for fn, dets, masks in results:
                for d in dets: tracker.add_detection(fn, d[0], d[1], d[2])
                if debug_writer:
                    orig_frame = next((b[1] for b in batch if b[0] == fn), None)
                    if orig_frame is not None:
                        debug_writer.write(visualizer.draw_debug_detections(orig_frame, fn, dets, roi_map))
                
    reader.stop()
    if mser_writer: mser_writer.release()
    if debug_writer: debug_writer.release()
    
    tracker.interpolate_and_smooth()
    
    # --- PASS 2: RECOGNIZE ---
    print("\n⚡ PASS 2: RECOGNIZE + RENDER")
    reader = ParallelFrameReader(config.INPUT_VIDEO_PATH, config.MAX_FRAME_ID, config.PREFETCH_FRAMES)
    reader.start()
    
    with ThreadPoolExecutor(max_workers=4) as exc:
        while True:
            item = reader.get()
            if item is None: break
            fnum, frame = item
            
            candidates = tracker.get_smoothed_detections(fnum)
            final_dets = recognizer.recognize_batch(frame, candidates)
            
            # Problem Sign Logic (Cấm đỗ xe)
            processed_dets = []
            red_boxes = [d[0] for d in final_dets if d[1]=='red' and d[2].get('sign_name') in config.PROBLEM_SIGNS]
            
            for bbox, color, metrics in final_dets:
                name = metrics.get('sign_name')
                if name in config.PROBLEM_SIGNS and color == 'blue':
                    overlap = False
                    for rb in red_boxes:
                        if calculate_iou(bbox, rb) > config.BBOX_OVERLAP_IOU: overlap = True; break
                    if not overlap:
                        bbox = expand_bbox(bbox, config.BBOX_EXPAND_SCALE, frame.shape)
                        processed_dets.append((bbox, color, metrics))
                else:
                    processed_dets.append((bbox, color, metrics))
            
            out = visualizer.draw(frame, fnum, processed_dets)
            video_writer.write(out)
            print(f"Rendered frame {fnum}", end='\r')

    reader.stop()
    video_writer.release()
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()