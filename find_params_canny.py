import cv2 
import numpy as np 
import time
from typing import List, Tuple, Dict
import os

# Fix Qt threading warning
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

# =============================================================================
# CLASS 1: CONFIGURATION (Sử dụng CANNY)
# =============================================================================
class TrafficSignConfig:
    def __init__(self):
        # --- File paths ---
        self.INPUT_VIDEO_PATH = 'videos/video2.mp4'
        self.STUDENT_IDS = "523H0164_523H0177_523H0145"
        
        # --- Performance settings ---
        self.PROCESS_SCALE = 1.0
        self.DISPLAY_SCALE = 0.5
        
        # --- Visualization ---
        self.DEBUG_MODE = True
        self.BOX_COLOR = (0, 255, 0)
        
        # === (MỚI) CẤU HÌNH CANNY EDGE DETECTION ===
        self.CANNY_BLUR_KSIZE = 5     # (Tune: 3, 5, 7)
        self.CANNY_THRESHOLD_1 = 50   # (Tune: 30-100)
        self.CANNY_THRESHOLD_2 = 150  # (Tune: 100-200)
        self.CANNY_MORPH_KSIZE = 3    # (Tune: 3, 5)
        self.CANNY_MORPH_ITER = 2     # (Tune: 1, 2, 3)
        
        # --- (MỚI) CẤU HÌNH ĐOÁN MÀU (YCrCb-MEAN) ---
        # Chúng ta vẫn cần đoán màu cho logic "Problem Signs"
        self.COLOR_GUESS_CR_MIN = 135 
        self.COLOR_GUESS_CB_MIN = 135 
        self.COLOR_GUESS_CR_MAX_FOR_YELLOW = 130
        self.COLOR_GUESS_CB_MAX_FOR_YELLOW = 120

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


# =============================================================================
# CLASS 2: OPTIMIZED DETECTOR (Sử dụng CANNY)
# =============================================================================
class TrafficSignDetector:
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        
    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(MỚI) Preprocessing: Gray -> Blur -> Canny VÀ YCrCb
        Trả về: (edge_mask, frame_ycrcb)
        """
        ksize = self.config.CANNY_BLUR_KSIZE
        if ksize % 2 == 0: ksize = max(1, ksize - 1) # Đảm bảo lẻ & > 0
        thresh1 = self.config.CANNY_THRESHOLD_1
        thresh2 = self.config.CANNY_THRESHOLD_2

        # 1. Tạo Edge Mask
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (ksize, ksize), 0)
        edge_mask = cv2.Canny(frame_blur, thresh1, thresh2)
        
        # 2. Tạo YCrCb frame (để đoán màu)
        frame_blur_bgr = cv2.medianBlur(frame, ksize)
        frame_ycrcb = cv2.cvtColor(frame_blur_bgr, cv2.COLOR_BGR2YCrCb)
        
        return edge_mask, frame_ycrcb

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """(MỚI) Nối các cạnh Canny bị đứt"""
        ksize = self.config.CANNY_MORPH_KSIZE
        if ksize % 2 == 0: ksize = max(1, ksize - 1) # Đảm bảo lẻ & > 0
        iters = self.config.CANNY_MORPH_ITER
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask_processed = cv2.dilate(mask, kernel, iterations=iters)
        return mask_processed
    
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

    def process_frame(self, frame: np.ndarray, full_frame_dims: Tuple) -> Tuple[List[Tuple], Dict[str, np.ndarray]]:
        """
        (MỚI) Pipeline dựa trên Canny Edge + Shape Filter + Color Guess
        Returns: (all_detections, masks_dict)
        """
        all_detections = []
        masks_dict = {}

        # 1. Tạo Edge Mask VÀ YCrCb Frame
        edge_mask, frame_ycrcb = self._preprocess_frame(frame)
        
        # 2. Xử lý Mask Cạnh
        mask_processed = self._apply_morphology(edge_mask)
        masks_dict['canny'] = mask_processed # Lưu mask canny để debug
        
        # 3. Tìm Kontua
        contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        circle_cfg = self.config.SHAPE_PARAMS['circle']
        triangle_cfg = self.config.SHAPE_PARAMS['triangle']
        
        # 4. Lọc Hình Dạng
        for contour in contours:
            area = cv2.contourArea(contour)
            bbox = cv2.boundingRect(contour)
            x, y, w, h = bbox
            if w <= 0 or h <= 0: continue

            # Thử lọc hình tròn
            if (area >= circle_cfg['min_area'] and area <= circle_cfg['max_area']):
                is_valid, metrics = self._detect_circles(contour, area, 
                                                        circle_cfg['trust_threshold'])
                if is_valid:
                    roi_ycrcb = frame_ycrcb[y:y+h, x:x+w]
                    color = self._get_dominant_color(roi_ycrcb)
                    all_detections.append((bbox, color, metrics))
                    continue 

            # Thử lọc hình tam giác
            if (area >= triangle_cfg['min_area'] and area <= triangle_cfg['max_area']):
                is_valid, metrics = self._detect_triangles(contour, area)
                if is_valid:
                    roi_ycrcb = frame_ycrcb[y:y+h, x:x+w]
                    color = self._get_dominant_color(roi_ycrcb)
                    all_detections.append((bbox, color, metrics))
        
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
            
            # (MỚI) Vẽ màu BBox dựa trên màu đã đoán
            if color_type == 'blue': box_color = (255, 0, 0)
            elif color_type == 'red': box_color = (0, 0, 255)
            elif color_type == 'yellow': box_color = (0, 255, 255)
            else: box_color = (128, 128, 128) # Màu xám cho 'unknown'

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
            
            if self.config.DEBUG_MODE and metrics:
                shape = metrics.get('shape', 'unknown')
                area = metrics.get('area', 0)
                
                if shape == 'circle': text = f"C:{metrics.get('circularity', 0):.2f}"
                else: text = f"S:{metrics.get('solidity', 0):.2f}"
                text = f"{color_type} A:{area} {text}" # Hiển thị màu đã đoán
                
                text_y = max(y - 10, 20)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, text_y - text_h - 5), 
                            (x + text_w + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame


# =============================================================================
# (Không thay đổi) UTILITY FUNCTIONS (is_bbox_in_roi, convert_roi_to_pixels)
# =============================================================================
def convert_roi_to_pixels(roi_percent: Tuple, w_full: int, h_full: int) -> Tuple:
    return (0,0,0,0) # Không dùng nữa

def is_bbox_in_roi(bbox: Tuple, roi_params: Tuple, overlap_threshold: float = 0.5) -> bool:
    return True # Không dùng nữa


# =============================================================================
# TRACKBAR CONTROL (Sử dụng CANNY)
# =============================================================================
class TrackbarController:
    """Manages trackbars for real-time Canny/Shape parameter tuning"""
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        self.window_name = "Parameter Controls (Canny)"
        
    def create_trackbars(self):
        """Create all trackbars for parameter tuning"""
        cv2.namedWindow(self.window_name)
        cv2.resizeWindow(self.window_name, 500, 500) # Cửa sổ nhỏ hơn
        
        # === CANNY PARAMETERS ===
        cv2.createTrackbar('Canny Blur', self.window_name, 
                          self.config.CANNY_BLUR_KSIZE, 15, lambda x: None)
        cv2.createTrackbar('Canny Thresh1', self.window_name, 
                          self.config.CANNY_THRESHOLD_1, 255, lambda x: None)
        cv2.createTrackbar('Canny Thresh2', self.window_name, 
                          self.config.CANNY_THRESHOLD_2, 255, lambda x: None)
        cv2.createTrackbar('Canny Morph KSize', self.window_name, 
                          self.config.CANNY_MORPH_KSIZE, 15, lambda x: None)
        cv2.createTrackbar('Canny Morph Iter', self.window_name, 
                          self.config.CANNY_MORPH_ITER, 10, lambda x: None)

        # === SHAPE PARAMETERS ===
        cv2.createTrackbar('Circle Circ x100', self.window_name, 
                            int(self.config.SHAPE_PARAMS['circle']['small_circularity'] * 100), 
                            100, lambda x: None)
        cv2.createTrackbar('Triangle Sol x100', self.window_name, 
                            int(self.config.SHAPE_PARAMS['triangle']['min_solidity'] * 100), 
                            100, lambda x: None)
        
        # === COLOR GUESS PARAMETERS (Ít khi tune) ===
        cv2.createTrackbar('Color Cr_min (Red)', self.window_name, 
                            self.config.COLOR_GUESS_CR_MIN, 255, lambda x: None)
        cv2.createTrackbar('Color Cb_min (Blue)', self.window_name, 
                            self.config.COLOR_GUESS_CB_MIN, 255, lambda x: None)
    
    def update_params(self, verbose=False):
        """Read trackbar values and update config"""
        
        # --- Canny ---
        blur_size = cv2.getTrackbarPos('Canny Blur', self.window_name)
        if blur_size % 2 == 0: blur_size = max(1, blur_size - 1)
        self.config.CANNY_BLUR_KSIZE = blur_size
        
        self.config.CANNY_THRESHOLD_1 = cv2.getTrackbarPos('Canny Thresh1', self.window_name)
        self.config.CANNY_THRESHOLD_2 = cv2.getTrackbarPos('Canny Thresh2', self.window_name)
        
        morph_ksize = cv2.getTrackbarPos('Canny Morph KSize', self.window_name)
        if morph_ksize % 2 == 0: morph_ksize = max(1, morph_ksize - 1)
        self.config.CANNY_MORPH_KSIZE = morph_ksize
        
        self.config.CANNY_MORPH_ITER = cv2.getTrackbarPos('Canny Morph Iter', self.window_name)

        # --- Shape ---
        circ = cv2.getTrackbarPos('Circle Circ x100', self.window_name) / 100.0
        self.config.SHAPE_PARAMS['circle']['small_circularity'] = circ
        self.config.SHAPE_PARAMS['circle']['large_circularity'] = circ
            
        sol = cv2.getTrackbarPos('Triangle Sol x100', self.window_name) / 100.0
        self.config.SHAPE_PARAMS['triangle']['min_solidity'] = sol
        
        # --- Color Guess ---
        self.config.COLOR_GUESS_CR_MIN = cv2.getTrackbarPos('Color Cr_min (Red)', self.window_name)
        self.config.COLOR_GUESS_CB_MIN = cv2.getTrackbarPos('Color Cb_min (Blue)', self.window_name)
        
        # Luôn trả về True để đảm bảo detector được cập nhật nếu cần
        return True
    
    def get_mask_visibility(self):
        # Không còn dùng, nhưng để lại cho tương thích
        return {}


# =============================================================================
# (Xóa) USER INPUT FUNCTION
# =============================================================================
# Không cần chọn màu nữa

# =============================================================================
# (Xóa) UTILITY: COMBINE MASKS
# =============================================================================
# Không cần combine mask nữa


# =============================================================================
# MAIN FUNCTION (REAL-TIME CANNY)
# =============================================================================
def main():
    print("=" * 70)
    print("  TRAFFIC SIGN DETECTION - REAL-TIME PARAMETER TUNING (CANNY)")
    print("=" * 70)
    
    try:
        # 1. (Xóa) Không cần chọn màu
        
        # 2. Initialize
        config = TrafficSignConfig()
        visualizer = Visualizer(config)
        trackbar_ctrl = TrackbarController(config)
        
        # 3. Open video
        cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video '{config.INPUT_VIDEO_PATH}'"); return
        
        fps = cap.get(cv2.CAP_PROP_FPS);
        if fps == 0: fps = 30.0
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
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  'r': Restart video")
        print("  'v': Toggle verbose mode (show parameter changes)")
        print("  'q' or ESC: Quit")
        print("  Use trackbars to tune CANNY and SHAPE parameters")
        
        # 4. Create trackbars
        trackbar_ctrl.create_trackbars()
        
        # 5. Create display windows
        cv2.namedWindow('Output Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Output Video', w_display, h_display)
        cv2.moveWindow('Output Video', 50, 50)
        
        # (MỚI) Chỉ tạo 1 cửa sổ Canny Mask
        cv2.namedWindow('Canny Mask', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Canny Mask', w_display, h_display)
        cv2.moveWindow('Canny Mask', 50, 50 + h_display + 30)
        
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
        print("STARTING REAL-TIME PROCESSING (CANNY)...")
        print("=" * 70 + "\n")
        
        current_frame = None
        current_frame_resized = None
        
        while True:
            # Luôn cập nhật thông số từ trackbar
            # (Chúng ta tạo lại detector mỗi frame vì nó nhẹ và đảm bảo
            # config luôn được cập nhật cho _preprocess_frame)
            trackbar_ctrl.update_params(verbose=verbose_mode)
            detector = TrafficSignDetector(config) 
            
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
                
                # Visualize
                frame_output = visualizer.draw_all(frame_display_sized, frame_count, scaled_detections, current_fps)
                cv2.imshow('Output Video', frame_output)
                
                # Hiển thị Canny Mask
                canny_mask_display = cv2.resize(masks['canny'], (w_display, h_display))
                cv2.imshow('Canny Mask', canny_mask_display)
            
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            
            if key == ord('q') or key == 27: break
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