import cv2 
import numpy as np 
import time
from typing import List, Tuple, Dict
import os

# Fix Qt threading warning
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

# =============================================================================
# CLASS 1: CONFIGURATION (Sử dụng YCrCb)
# =============================================================================
class TrafficSignConfig:
    def __init__(self, selected_colors=None):
        # --- File paths ---
        self.INPUT_VIDEO_PATH = 'videos/video2.mp4'
        self.STUDENT_IDS = "523H0164_523H0177_523H0145"
        
        # --- Performance settings ---
        self.PROCESS_SCALE = 1.0
        self.DISPLAY_SCALE = 0.5
        
        # --- Visualization ---
        self.DEBUG_MODE = True
        self.BOX_COLOR = (0, 255, 0)
        
        # --- Selected colors for detection ---
        self.SELECTED_COLORS = selected_colors if selected_colors else ['blue', 'red', 'yellow']
        
        # --- Color-specific parameters (ĐÃ CHUYỂN SANG YCrCb) ---
        # Lưu ý: Đây là các dải màu GỢI Ý. Bạn PHẢI tự tinh chỉnh!
        # Y = Độ sáng (0-255)
        # Cr = Chênh lệch Đỏ-Xanh lá (0-255, <128 là xanh lá, >128 là đỏ)
        # Cb = Chênh lệch Xanh dương-Vàng (0-255, <128 là vàng, >128 là xanh)
        self.COLOR_PARAMS = {
            'blue': {
                'ycrcb_lower': np.array([0, 0, 130]),
                'ycrcb_upper': np.array([255, 125, 255]),
                'morph_ksize': 7, 'open_iter': 1, 'close_iter': 5,
                'blur_ksize': 7,
                'roi': (0.0, 0.0, 1.0, 1.0),
                'shape_type': 'circle'
            },
            'red': {
                'ycrcb_lower': np.array([0, 135, 0]),
                'ycrcb_upper': np.array([255, 255, 125]),
                'morph_ksize': 7, 'open_iter': 0, 'close_iter': 0,
                'blur_ksize': 5,
                'roi': (0.0, 0.0, 1.0, 1.0),
                'shape_type': 'circle'
            },
            'yellow': {
                'ycrcb_lower': np.array([100, 0, 0]), # Yêu cầu độ sáng tối thiểu
                'ycrcb_upper': np.array([255, 130, 120]), # Cr, Cb ở mức thấp/trung
                'morph_ksize': 3, 'open_iter': 1, 'close_iter': 5,
                'blur_ksize': 5,
                'roi': (0.0, 0.0, 1.0, 1.0),
                'shape_type': 'triangle'
            }
        }
        
        # --- Image processing (Đã xóa CLAHE/Saturation) ---
        
        # --- Shape detection parameters ---
        self.SHAPE_PARAMS = {
            'circle': {
                'min_area': 200, 'max_area': 15000,
                'trust_threshold': 725,
                'small_circularity': 0.80,
                'large_circularity': 0.85
            },
            'triangle': {
                'min_area': 200, 'max_area': 50000,
                'trust_threshold': 1500,
                'min_solidity': 0.75,
                'epsilon_factor': 0.03,
                'max_vertices': 7
            }
        }


# =============================================================================
# CLASS 2: OPTIMIZED DETECTOR (Sử dụng YCrCb)
# =============================================================================
class TrafficSignDetector:
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        # Đã xóa CLAHE
        
    def _preprocess_frame(self, frame: np.ndarray, blur_ksize: int) -> np.ndarray:
        """Preprocessing: blur + YCrCb conversion"""
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        
        frame_proc = cv2.medianBlur(frame, blur_ksize)
        # Chuyển đổi sang YCrCb thay vì HSV
        frame_proc = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2YCrCb)
        
        return frame_proc
    
    def _apply_morphology(self, mask: np.ndarray, k_size: int, 
                         open_iter: int, close_iter: int) -> np.ndarray:
        """Apply morphological operations (Giữ nguyên)"""
        if k_size <= 1:
            return mask
        if k_size % 2 == 0:
            k_size += 1
            
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_close,iterations=2)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.erode(mask, kernel_erode, iterations=2)

        return mask
    
    def _detect_circles(self, contour: np.ndarray, area: float, 
                       trust_threshold: float) -> Tuple[bool, Dict]:
        """Circle detection with circularity check (Giữ nguyên)"""
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
        """Triangle detection with solidity and vertex checks (Giữ nguyên)"""
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
    
    def _process_color(self, frame_ycrcb: np.ndarray, color_name: str, 
                      roi_params: Tuple) -> Tuple[List[Tuple], np.ndarray]:
        """
        Xử lý 1 màu: Lọc YCrCb + Morphology + Lọc hình dạng
        Returns: (detections, mask)
        """
        params = self.config.COLOR_PARAMS[color_name]
        
        # Color segmentation (SỬ DỤNG YCrCb)
        mask = cv2.inRange(frame_ycrcb, params['ycrcb_lower'], params['ycrcb_upper'])
        
        # Morphology
        mask_clean = self._apply_morphology(
            mask, params['morph_ksize'], 
            params['open_iter'], params['close_iter']
        )
        
        # Find contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Shape detection
        detections = []
        shape_type = params['shape_type']
        shape_cfg = self.config.SHAPE_PARAMS[shape_type]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < shape_cfg['min_area'] or area > shape_cfg['max_area']:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            is_in_roi = is_bbox_in_roi((x, y, w, h), roi_params, overlap_threshold=0.5)
            
            if area < shape_cfg['trust_threshold'] and not is_in_roi:
                continue
            
            if shape_type == 'circle':
                is_valid, metrics = self._detect_circles(contour, area, 
                                                        shape_cfg['trust_threshold'])
            else:  # triangle
                is_valid, metrics = self._detect_triangles(contour, area)
            
            if is_valid:
                detections.append(((x, y, w, h), color_name, metrics))
        
        return detections, mask_clean
    
    def process_frame(self, frame: np.ndarray, full_frame_dims: Tuple) -> Tuple[List[Tuple], Dict[str, np.ndarray]]:
        """
        (ĐÃ SỬA LỖI LOGIC)
        Xử lý frame bằng cách lặp qua từng màu,
        chạy pipeline đầy đủ (Blur, YCrCb, inRange, Morph, Shape)
        Returns: (all_detections, masks_dict)
        """
        w_full, h_full = full_frame_dims
        all_detections = []
        masks_dict = {}

        for color_name in self.config.SELECTED_COLORS:
            params = self.config.COLOR_PARAMS[color_name]
            
            # 1. Preprocess (Blur + Convert to YCrCb)
            # (Thực hiện cho mỗi màu để tôn trọng cài đặt blur_ksize riêng)
            frame_ycrcb = self._preprocess_frame(frame, params['blur_ksize'])
            
            # 2. Process Color (inRange + Morph + Shape Detect)
            roi_pix = convert_roi_to_pixels(params['roi'], w_full, h_full)
            detections, mask = self._process_color(frame_ycrcb, color_name, roi_pix)
            
            all_detections.extend(detections)
            masks_dict[color_name] = mask
        
        # Tạo mask trống cho các màu không được chọn
        for color_name in ['blue', 'red', 'yellow']:
            if color_name not in self.config.SELECTED_COLORS:
                masks_dict[color_name] = np.zeros((h_full, w_full), dtype=np.uint8)
        
        return all_detections, masks_dict


# =============================================================================
# CLASS 3: VISUALIZER (simplified for real-time)
# =============================================================================
class Visualizer:
    # (Không thay đổi)
    def __init__(self, config: TrafficSignConfig):
        self.config = config
    
    def draw_all(self, frame: np.ndarray, frame_num: int, 
                 detections: List[Tuple], fps: float = 0.0) -> np.ndarray:
        frame_output = frame.copy()
        frame_output = self._draw_student_ids(frame_output)
        frame_output = self._draw_frame_id(frame_output, frame_num)
        if fps > 0:
            frame_output = self._draw_fps(frame_output, fps)
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
    
    def _draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        text = f"FPS: {fps:.1f}"
        x, y = 10, 110
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 1.0, 2)
        cv2.rectangle(frame, (x - 5, y - text_h - 5), 
                     (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        for bbox, color_type, metrics in detections:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.config.BOX_COLOR, 3)
            
            if self.config.DEBUG_MODE and metrics:
                shape = metrics.get('shape', 'unknown')
                area = metrics.get('area', 0)
                
                if shape == 'circle':
                    text = f"{color_type}: A:{area} C:{metrics.get('circularity', 0):.2f}"
                else:
                    text = f"{color_type}: A:{area} S:{metrics.get('solidity', 0):.2f}"
                
                text_y = max(y - 10, 20)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, text_y - text_h - 5), 
                            (x + text_w + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame


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


# =============================================================================
# TRACKBAR CONTROL (Sử dụng YCrCb)
# =============================================================================
class TrackbarController:
    """Manages trackbars for real-time parameter tuning"""
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        self.window_name = "Parameter Controls (YCrCb)" # Đổi tên
        
    def create_trackbars(self):
        """Create all trackbars for parameter tuning"""
        cv2.namedWindow(self.window_name)
        cv2.resizeWindow(self.window_name, 500, 800)
        
        # === COMBINED MASK CONTROLS ===
        cv2.createTrackbar('Show Blue', self.window_name, 1, 1, lambda x: None)
        cv2.createTrackbar('Show Red', self.window_name, 1, 1, lambda x: None)
        cv2.createTrackbar('Show Yellow', self.window_name, 1, 1, lambda x: None)
        
        # === ĐÃ XÓA CLAHE VÀ SATURATION TRACKBARS ===
        
        # Blue parameters (Sử dụng YCrCb 0-255)
        if 'blue' in self.config.SELECTED_COLORS:
            cv2.createTrackbar('Blue Y_min', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['ycrcb_lower'][0], 255, lambda x: None)
            cv2.createTrackbar('Blue Y_max', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['ycrcb_upper'][0], 255, lambda x: None)
            cv2.createTrackbar('Blue Cr_min', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['ycrcb_lower'][1], 255, lambda x: None)
            cv2.createTrackbar('Blue Cr_max', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['ycrcb_upper'][1], 255, lambda x: None)
            cv2.createTrackbar('Blue Cb_min', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['ycrcb_lower'][2], 255, lambda x: None)
            cv2.createTrackbar('Blue Cb_max', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['ycrcb_upper'][2], 255, lambda x: None)
            # Morphology Parameters
            cv2.createTrackbar('Blue Morph Size', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['morph_ksize'], 15, lambda x: None)
            cv2.createTrackbar('Blue Open Iter', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['open_iter'], 10, lambda x: None)
            cv2.createTrackbar('Blue Close Iter', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['close_iter'], 10, lambda x: None)
            cv2.createTrackbar('Blue Blur Size', self.window_name, 
                              self.config.COLOR_PARAMS['blue']['blur_ksize'], 15, lambda x: None)
        
        # Red parameters (Sử dụng YCrCb 0-255)
        if 'red' in self.config.SELECTED_COLORS:
            cv2.createTrackbar('Red Y_min', self.window_name, 
                              self.config.COLOR_PARAMS['red']['ycrcb_lower'][0], 255, lambda x: None)
            cv2.createTrackbar('Red Y_max', self.window_name, 
                              self.config.COLOR_PARAMS['red']['ycrcb_upper'][0], 255, lambda x: None)
            cv2.createTrackbar('Red Cr_min', self.window_name, 
                              self.config.COLOR_PARAMS['red']['ycrcb_lower'][1], 255, lambda x: None)
            cv2.createTrackbar('Red Cr_max', self.window_name, 
                              self.config.COLOR_PARAMS['red']['ycrcb_upper'][1], 255, lambda x: None)
            cv2.createTrackbar('Red Cb_min', self.window_name, 
                              self.config.COLOR_PARAMS['red']['ycrcb_lower'][2], 255, lambda x: None)
            cv2.createTrackbar('Red Cb_max', self.window_name, 
                              self.config.COLOR_PARAMS['red']['ycrcb_upper'][2], 255, lambda x: None)
            # Morphology Parameters
            cv2.createTrackbar('Red Morph Size', self.window_name, 
                              self.config.COLOR_PARAMS['red']['morph_ksize'], 15, lambda x: None)
            cv2.createTrackbar('Red Open Iter', self.window_name, 
                              self.config.COLOR_PARAMS['red']['open_iter'], 10, lambda x: None)
            cv2.createTrackbar('Red Close Iter', self.window_name, 
                              self.config.COLOR_PARAMS['red']['close_iter'], 10, lambda x: None)
            cv2.createTrackbar('Red Blur Size', self.window_name, 
                              self.config.COLOR_PARAMS['red']['blur_ksize'], 15, lambda x: None)
        
        # Yellow parameters (Sử dụng YCrCb 0-255)
        if 'yellow' in self.config.SELECTED_COLORS:
            cv2.createTrackbar('Yellow Y_min', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['ycrcb_lower'][0], 255, lambda x: None)
            cv2.createTrackbar('Yellow Y_max', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['ycrcb_upper'][0], 255, lambda x: None)
            cv2.createTrackbar('Yellow Cr_min', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['ycrcb_lower'][1], 255, lambda x: None)
            cv2.createTrackbar('Yellow Cr_max', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['ycrcb_upper'][1], 255, lambda x: None)
            cv2.createTrackbar('Yellow Cb_min', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['ycrcb_lower'][2], 255, lambda x: None)
            cv2.createTrackbar('Yellow Cb_max', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['ycrcb_upper'][2], 255, lambda x: None)
            # Morphology Parameters
            cv2.createTrackbar('Yellow Morph Size', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['morph_ksize'], 15, lambda x: None)
            cv2.createTrackbar('Yellow Open Iter', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['open_iter'], 10, lambda x: None)
            cv2.createTrackbar('Yellow Close Iter', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['close_iter'], 10, lambda x: None)
            cv2.createTrackbar('Yellow Blur Size', self.window_name, 
                              self.config.COLOR_PARAMS['yellow']['blur_ksize'], 15, lambda x: None)
        
        # Shape parameters (Giữ nguyên)
        if any(self.config.COLOR_PARAMS[c]['shape_type'] == 'circle' for c in self.config.SELECTED_COLORS):
            cv2.createTrackbar('Circle Circ x100', self.window_name, 
                              int(self.config.SHAPE_PARAMS['circle']['small_circularity'] * 100), 
                              100, lambda x: None)
        if any(self.config.COLOR_PARAMS[c]['shape_type'] == 'triangle' for c in self.config.SELECTED_COLORS):
            cv2.createTrackbar('Triangle Sol x100', self.window_name, 
                              int(self.config.SHAPE_PARAMS['triangle']['min_solidity'] * 100), 
                              100, lambda x: None)
    
    def update_params(self, verbose=False):
        """Read trackbar values and update config"""
        # Đã xóa CLAHE và Saturation
        
        # Blue (Sử dụng YCrCb)
        if 'blue' in self.config.SELECTED_COLORS:
            y_min = cv2.getTrackbarPos('Blue Y_min', self.window_name)
            y_max = cv2.getTrackbarPos('Blue Y_max', self.window_name)
            cr_min = cv2.getTrackbarPos('Blue Cr_min', self.window_name)
            cr_max = cv2.getTrackbarPos('Blue Cr_max', self.window_name)
            cb_min = cv2.getTrackbarPos('Blue Cb_min', self.window_name)
            cb_max = cv2.getTrackbarPos('Blue Cb_max', self.window_name)
            
            new_lower = np.array([y_min, cr_min, cb_min])
            new_upper = np.array([y_max, cr_max, cb_max])
            
            if not np.array_equal(new_lower, self.config.COLOR_PARAMS['blue']['ycrcb_lower']) or \
               not np.array_equal(new_upper, self.config.COLOR_PARAMS['blue']['ycrcb_upper']):
                if verbose:
                    print(f"Blue YCrCb changed: Y[{y_min}-{y_max}] Cr[{cr_min}-{cr_max}] Cb[{cb_min}-{cb_max}]")
            
            self.config.COLOR_PARAMS['blue']['ycrcb_lower'] = new_lower
            self.config.COLOR_PARAMS['blue']['ycrcb_upper'] = new_upper
            
            # Morphology (Giữ nguyên)
            morph_size = cv2.getTrackbarPos('Blue Morph Size', self.window_name)
            open_iter = cv2.getTrackbarPos('Blue Open Iter', self.window_name)
            close_iter = cv2.getTrackbarPos('Blue Close Iter', self.window_name)
            blur_size = cv2.getTrackbarPos('Blue Blur Size', self.window_name)
            
            if morph_size % 2 == 0: morph_size = max(1, morph_size - 1)
            if blur_size % 2 == 0: blur_size = max(1, blur_size - 1)
            
            self.config.COLOR_PARAMS['blue']['morph_ksize'] = morph_size
            self.config.COLOR_PARAMS['blue']['open_iter'] = open_iter
            self.config.COLOR_PARAMS['blue']['close_iter'] = close_iter
            self.config.COLOR_PARAMS['blue']['blur_ksize'] = blur_size
        
        # Red (Sử dụng YCrCb)
        if 'red' in self.config.SELECTED_COLORS:
            y_min = cv2.getTrackbarPos('Red Y_min', self.window_name)
            y_max = cv2.getTrackbarPos('Red Y_max', self.window_name)
            cr_min = cv2.getTrackbarPos('Red Cr_min', self.window_name)
            cr_max = cv2.getTrackbarPos('Red Cr_max', self.window_name)
            cb_min = cv2.getTrackbarPos('Red Cb_min', self.window_name)
            cb_max = cv2.getTrackbarPos('Red Cb_max', self.window_name)
            
            new_lower = np.array([y_min, cr_min, cb_min])
            new_upper = np.array([y_max, cr_max, cb_max])
            
            if not np.array_equal(new_lower, self.config.COLOR_PARAMS['red']['ycrcb_lower']) or \
               not np.array_equal(new_upper, self.config.COLOR_PARAMS['red']['ycrcb_upper']):
                if verbose:
                    print(f"Red YCrCb changed: Y[{y_min}-{y_max}] Cr[{cr_min}-{cr_max}] Cb[{cb_min}-{cb_max}]")
            
            self.config.COLOR_PARAMS['red']['ycrcb_lower'] = new_lower
            self.config.COLOR_PARAMS['red']['ycrcb_upper'] = new_upper
            
            # Morphology (Giữ nguyên)
            morph_size = cv2.getTrackbarPos('Red Morph Size', self.window_name)
            open_iter = cv2.getTrackbarPos('Red Open Iter', self.window_name)
            close_iter = cv2.getTrackbarPos('Red Close Iter', self.window_name)
            blur_size = cv2.getTrackbarPos('Red Blur Size', self.window_name)
            
            if morph_size % 2 == 0: morph_size = max(1, morph_size - 1)
            if blur_size % 2 == 0: blur_size = max(1, blur_size - 1)
            
            self.config.COLOR_PARAMS['red']['morph_ksize'] = morph_size
            self.config.COLOR_PARAMS['red']['open_iter'] = open_iter
            self.config.COLOR_PARAMS['red']['close_iter'] = close_iter
            self.config.COLOR_PARAMS['red']['blur_ksize'] = blur_size
        
        # Yellow (Sử dụng YCrCb)
        if 'yellow' in self.config.SELECTED_COLORS:
            y_min = cv2.getTrackbarPos('Yellow Y_min', self.window_name)
            y_max = cv2.getTrackbarPos('Yellow Y_max', self.window_name)
            cr_min = cv2.getTrackbarPos('Yellow Cr_min', self.window_name)
            cr_max = cv2.getTrackbarPos('Yellow Cr_max', self.window_name)
            cb_min = cv2.getTrackbarPos('Yellow Cb_min', self.window_name)
            cb_max = cv2.getTrackbarPos('Yellow Cb_max', self.window_name)
            
            new_lower = np.array([y_min, cr_min, cb_min])
            new_upper = np.array([y_max, cr_max, cb_max])
            
            if not np.array_equal(new_lower, self.config.COLOR_PARAMS['yellow']['ycrcb_lower']) or \
               not np.array_equal(new_upper, self.config.COLOR_PARAMS['yellow']['ycrcb_upper']):
                if verbose:
                    print(f"Yellow YCrCb changed: Y[{y_min}-{y_max}] Cr[{cr_min}-{cr_max}] Cb[{cb_min}-{cb_max}]")
            
            self.config.COLOR_PARAMS['yellow']['ycrcb_lower'] = new_lower
            self.config.COLOR_PARAMS['yellow']['ycrcb_upper'] = new_upper
            
            # Morphology (Giữ nguyên)
            morph_size = cv2.getTrackbarPos('Yellow Morph Size', self.window_name)
            open_iter = cv2.getTrackbarPos('Yellow Open Iter', self.window_name)
            close_iter = cv2.getTrackbarPos('Yellow Close Iter', self.window_name)
            blur_size = cv2.getTrackbarPos('Yellow Blur Size', self.window_name)
            
            if morph_size % 2 == 0: morph_size = max(1, morph_size - 1)
            if blur_size % 2 == 0: blur_size = max(1, blur_size - 1)
            
            self.config.COLOR_PARAMS['yellow']['morph_ksize'] = morph_size
            self.config.COLOR_PARAMS['yellow']['open_iter'] = open_iter
            self.config.COLOR_PARAMS['yellow']['close_iter'] = close_iter
            self.config.COLOR_PARAMS['yellow']['blur_ksize'] = blur_size
        
        # Shape parameters (Giữ nguyên)
        if any(self.config.COLOR_PARAMS[c]['shape_type'] == 'circle' for c in self.config.SELECTED_COLORS):
            circ = cv2.getTrackbarPos('Circle Circ x100', self.window_name) / 100.0
            if abs(circ - self.config.SHAPE_PARAMS['circle']['small_circularity']) > 0.01:
                if verbose:
                    print(f"Circle circularity changed: {self.config.SHAPE_PARAMS['circle']['small_circularity']:.2f} -> {circ:.2f}")
            self.config.SHAPE_PARAMS['circle']['small_circularity'] = circ
            self.config.SHAPE_PARAMS['circle']['large_circularity'] = circ
            
        if any(self.config.COLOR_PARAMS[c]['shape_type'] == 'triangle' for c in self.config.SELECTED_COLORS):
            sol = cv2.getTrackbarPos('Triangle Sol x100', self.window_name) / 100.0
            if abs(sol - self.config.SHAPE_PARAMS['triangle']['min_solidity']) > 0.01:
                if verbose:
                    print(f"Triangle solidity changed: {self.config.SHAPE_PARAMS['triangle']['min_solidity']:.2f} -> {sol:.2f}")
            self.config.SHAPE_PARAMS['triangle']['min_solidity'] = sol
        
        # Trả về True nếu có thay đổi (luôn True để đơn giản, vì đã xóa CLAHE)
        return True
    
    def get_mask_visibility(self):
        """Get which masks should be shown in combined view"""
        return {
            'blue': cv2.getTrackbarPos('Show Blue', self.window_name) == 1,
            'red': cv2.getTrackbarPos('Show Red', self.window_name) == 1,
            'yellow': cv2.getTrackbarPos('Show Yellow', self.window_name) == 1
        }


# =============================================================================
# USER INPUT FUNCTION
# =============================================================================
def get_color_selection():
    # (Không thay đổi)
    print("\n" + "=" * 70)
    print("  COLOR SELECTION")
    print("=" * 70)
    print("\nAvailable colors:")
    print("  1. Blue (circles)")
    print("  2. Red (circles)")
    print("  3. Yellow (triangles)")
    print("  4. All colors")
    print()
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            return ['blue']
        elif choice == '2':
            return ['red']
        elif choice == '3':
            return ['yellow']
        elif choice == '4':
            return ['blue', 'red', 'yellow']
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


# =============================================================================
# UTILITY: COMBINE MASKS
# =============================================================================
def create_combined_mask(masks, visibility, selected_colors):
    # (Không thay đổi)
    h, w = None, None
    for color in selected_colors:
        if color in masks:
            h, w = masks[color].shape
            break
    
    if h is None:
        # Tìm một mask bất kỳ để lấy kích thước
        for mask in masks.values():
            if mask is not None:
                h, w = mask.shape
                break
        if h is None:
             return np.zeros((480, 640), dtype=np.uint8) # Fallback
    
    combined = np.zeros((h, w), dtype=np.uint8)
    
    if 'blue' in selected_colors and visibility.get('blue', True):
        combined = cv2.bitwise_or(combined, masks['blue'])
    
    if 'red' in selected_colors and visibility.get('red', True):
        combined = cv2.bitwise_or(combined, masks['red'])
    
    if 'yellow' in selected_colors and visibility.get('yellow', True):
        combined = cv2.bitwise_or(combined, masks['yellow'])
    
    return combined


# =============================================================================
# MAIN FUNCTION (REAL-TIME VERSION)
# =============================================================================
def main():
    print("=" * 70)
    print("  TRAFFIC SIGN DETECTION - REAL-TIME PARAMETER TUNING (YCrCb)")
    print("=" * 70)
    
    try:
        # 1. Get color selection from user
        selected_colors = get_color_selection()
        color_names = ', '.join([c.capitalize() for c in selected_colors])
        print(f"\n✓ Selected colors: {color_names}")
        
        # 2. Initialize with selected colors
        config = TrafficSignConfig(selected_colors=selected_colors)
        visualizer = Visualizer(config)
        trackbar_ctrl = TrackbarController(config)
        
        # 3. Open video
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
        
        w_process = int(w_orig * config.PROCESS_SCALE)
        h_process = int(h_orig * config.PROCESS_SCALE)
        w_display = int(w_orig * config.DISPLAY_SCALE)
        h_display = int(h_orig * config.DISPLAY_SCALE)
        full_frame_dims = (w_process, h_process)
        
        print(f"\nVideo: {config.INPUT_VIDEO_PATH}")
        print(f"Original: {w_orig}x{h_orig} @ {fps:.2f} FPS")
        print(f"Processing: {w_process}x{h_process} ({int(config.PROCESS_SCALE*100)}%)")
        print(f"Display: {w_display}x{h_display} ({int(config.DISPLAY_SCALE*100)}%)")
        print(f"Total frames: {total_frames}")
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  'r': Restart video")
        print("  'v': Toggle verbose mode (show parameter changes)")
        print("  'q' or ESC: Quit")
        print("  Use trackbars to tune YCrCb parameters in real-time")
        
        # 4. Create trackbars
        trackbar_ctrl.create_trackbars()
        
        # 5. Create display windows
        cv2.namedWindow('Output Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Output Video', w_display, h_display)
        cv2.moveWindow('Output Video', 50, 50)
        
        cv2.namedWindow('Combined Mask', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Combined Mask', w_display, h_display)
        cv2.moveWindow('Combined Mask', 50, 50 + h_display + 30)
        
        mask_width = w_display // 2
        mask_height = h_display // 2
        mask_x_offset = 50 + w_display + 20
        mask_y_pos = 50
        
        if 'blue' in selected_colors:
            cv2.namedWindow('Blue Mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Blue Mask', mask_width, mask_height)
            cv2.moveWindow('Blue Mask', mask_x_offset, mask_y_pos)
            mask_y_pos += mask_height + 30
        if 'red' in selected_colors:
            cv2.namedWindow('Red Mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Red Mask', mask_width, mask_height)
            cv2.moveWindow('Red Mask', mask_x_offset, mask_y_pos)
            mask_y_pos += mask_height + 30
        if 'yellow' in selected_colors:
            cv2.namedWindow('Yellow Mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Yellow Mask', mask_width, mask_height)
            cv2.moveWindow('Yellow Mask', mask_x_offset, mask_y_pos)
        
        cv2.waitKey(100)
        
        # 6. Real-time processing loop
        detector = TrafficSignDetector(config)
        frame_count = 0
        paused = False
        verbose_mode = False
        
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        
        print("\n" + "=" * 70)
        print("STARTING REAL-TIME PROCESSING (YCrCb)...")
        print("=" * 70 + "\n")
        
        current_frame = None
        current_frame_resized = None
        
        while True:
            # Luôn cập nhật thông số từ trackbar
            trackbar_ctrl.update_params(verbose=verbose_mode)
            # Vì detector không còn state (CLAHE), chúng ta không cần tạo lại nó
            
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    continue
                
                fps_frame_count += 1
                if fps_frame_count >= 10:
                    elapsed = time.time() - fps_start_time
                    current_fps = fps_frame_count / elapsed if elapsed > 0 else 0
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                current_frame = frame.copy()
                current_frame_resized = cv2.resize(frame, (w_process, h_process))
                frame_count += 1
            
            if current_frame_resized is not None:
                # Process resized frame
                detections, masks = detector.process_frame(current_frame_resized, full_frame_dims)
                
                # Scale detections back to display size
                scale_factor = config.DISPLAY_SCALE / config.PROCESS_SCALE
                scaled_detections = []
                for bbox, color, metrics in detections:
                    x, y, w, h = bbox
                    scaled_bbox = (
                        int(x * scale_factor),
                        int(y * scale_factor),
                        int(w * scale_factor),
                        int(h * scale_factor)
                    )
                    scaled_detections.append((scaled_bbox, color, metrics))
                
                frame_display_sized = cv2.resize(current_frame, (w_display, h_display))
                
                frame_output = visualizer.draw_all(frame_display_sized, frame_count, scaled_detections, current_fps)
                
                cv2.imshow('Output Video', frame_output)
                
                mask_visibility = trackbar_ctrl.get_mask_visibility()
                
                combined_mask = create_combined_mask(masks, mask_visibility, selected_colors)
                combined_mask_display = cv2.resize(combined_mask, (w_display, h_display))
                cv2.imshow('Combined Mask', combined_mask_display)
                
                if 'blue' in selected_colors:
                    blue_mask = cv2.resize(masks['blue'], (mask_width, mask_height))
                    cv2.imshow('Blue Mask', blue_mask)
                if 'red' in selected_colors:
                    red_mask = cv2.resize(masks['red'], (mask_width, mask_height))
                    cv2.imshow('Red Mask', red_mask)
                if 'yellow' in selected_colors:
                    yellow_mask = cv2.resize(masks['yellow'], (mask_width, mask_height))
                    cv2.imshow('Yellow Mask', yellow_mask)
            
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                print("Video restarted")
            elif key == ord('v'):
                verbose_mode = not verbose_mode
                print(f"Verbose mode: {'ON' if verbose_mode else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("✓ PROGRAM TERMINATED")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":
    main()