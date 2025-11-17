import cv2 
import numpy as np 
import time
from typing import List, Tuple, Dict
import os

# Fix Qt threading warning
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

# =============================================================================
# CLASS 1: CONFIGURATION (Sử dụng MSER)
# =============================================================================
class TrafficSignConfig:
    def __init__(self):
        # --- File paths ---
        self.INPUT_VIDEO_PATH = 'videos/video1.mp4'
        self.STUDENT_IDS = "523H0164_523H0177_523H0145"
        
        # --- Performance settings ---
        self.PROCESS_SCALE = 1.0
        self.DISPLAY_SCALE = 0.5
        
        # --- Visualization ---
        self.DEBUG_MODE = True
        self.BOX_COLOR = (0, 255, 0)
        
        # === (MỚI) CẤU HÌNH MSER (Sẽ được dùng cho cả 2 kênh Cr và Cb) ===
        self.MSER_DELTA = 5          # (Tune 1-20) 
        self.MSER_MIN_AREA = 60      # (Tune 30-200) 
        self.MSER_MAX_AREA = 14400   # (Tune 5000-30000)
        self.MSER_MAX_VARIATION = 0.25 # (Tune 0.1 - 1.0)
        self.MSER_MIN_DIVERSITY = 0.2  # (Tune 0.1 - 1.0)

        # --- (GIỮ NGUYÊN) CẤU HÌNH ĐOÁN MÀU ---
        self.COLOR_GUESS_CR_MIN = 135 
        self.COLOR_GUESS_CB_MIN = 125
        self.COLOR_GUESS_CR_MAX_FOR_YELLOW = 130
        self.COLOR_GUESS_CB_MAX_FOR_YELLOW = 120

        # --- (GIỮ NGUYÊN) Shape detection parameters ---
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
# CLASS 2: OPTIMIZED DETECTOR (Sử dụng MSER trên kênh Cr và Cb)
# =============================================================================
class TrafficSignDetector:
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        
        # (MỚI) Tạo 2 MSER instance, CÙNG MỘT CẤU HÌNH
        # Một cho kênh Cr (Đỏ/Vàng)
        self.mser_cr = cv2.MSER_create(
            delta=self.config.MSER_DELTA,
            min_area=self.config.MSER_MIN_AREA,
            max_area=self.config.MSER_MAX_AREA,
            max_variation=self.config.MSER_MAX_VARIATION,
            min_diversity=self.config.MSER_MIN_DIVERSITY
        )
        # Một cho kênh Cb (Xanh/Vàng)
        self.mser_cb = cv2.MSER_create(
            delta=self.config.MSER_DELTA,
            min_area=self.config.MSER_MIN_AREA,
            max_area=self.config.MSER_MAX_AREA,
            max_variation=self.config.MSER_MAX_VARIATION,
            min_diversity=self.config.MSER_MIN_DIVERSITY
        )
        
    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(MỚI) Preprocessing: Tách kênh YCrCb
        Trả về: (Y, Cr, Cb, frame_ycrcb_full)
        """
        # Blur nhẹ để giảm nhiễu cho MSER
        frame_blur_bgr = cv2.medianBlur(frame, 5)
        
        # 1. Tạo YCrCb frame 
        frame_ycrcb = cv2.cvtColor(frame_blur_bgr, cv2.COLOR_BGR2YCrCb)
        
        # 2. Tách các kênh
        Y, Cr, Cb = cv2.split(frame_ycrcb)
        
        return Y, Cr, Cb, frame_ycrcb
    
    def _get_dominant_color(self, roi_ycrcb: np.ndarray) -> str:
        """(GIỮ NGUYÊN) Đoán màu dựa trên YCrCb trung bình của ROI"""
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

    # (GIỮ NGUYÊN) _detect_circles và _detect_triangles
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
        (MỚI) Pipeline dựa trên MSER (Dual Channel) + Shape Filter + Color Guess
        """
        all_detections = []
        masks_dict = {}

        # 1. Tách các kênh Y, Cr, Cb
        Y, Cr, Cb, frame_ycrcb = self._preprocess_frame(frame)
        
        # 2. Chạy MSER trên kênh Cr (tìm Đỏ/Vàng)
        regions_cr, bboxes_cr = self.mser_cr.detectRegions(Cr)
        
        # 3. Chạy MSER trên kênh Cb (tìm Xanh)
        regions_cb, bboxes_cb = self.mser_cb.detectRegions(Cb)
        
        # 4. (MỚI) Kết hợp kết quả
        all_regions = regions_cr + regions_cb
        all_bboxes = np.vstack((bboxes_cr, bboxes_cb)) if len(bboxes_cr) > 0 and len(bboxes_cb) > 0 else \
                     (bboxes_cr if len(bboxes_cr) > 0 else (bboxes_cb if len(bboxes_cb) > 0 else []))

        # (MỚI) Tạo MSER mask để hiển thị (từ cả 2 kênh)
        mser_mask = np.zeros_like(Y)
        if all_regions:
            cv2.drawContours(mser_mask, all_regions, -1, (255), 1)
        masks_dict['mser_dual'] = mser_mask # Đổi tên mask

        circle_cfg = self.config.SHAPE_PARAMS['circle']
        triangle_cfg = self.config.SHAPE_PARAMS['triangle']
        
        # 5. Lọc Hình Dạng (trên TẤT CẢ các vùng tìm được)
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
                    all_detections.append((tuple(bbox), color, metrics)) # Chuyển bbox về tuple
                    continue 

            # Thử lọc hình tam giác
            if (area >= triangle_cfg['min_area'] and area <= triangle_cfg['max_area']):
                is_valid, metrics = self._detect_triangles(contour, area)
                if is_valid:
                    roi_ycrcb = frame_ycrcb[y:y+h, x:x+w]
                    color = self._get_dominant_color(roi_ycrcb)
                    all_detections.append((tuple(bbox), color, metrics)) # Chuyển bbox về tuple
        
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
            if color_type == 'blue': box_color = (255, 0, 0)
            elif color_type == 'red': box_color = (0, 0, 255)
            elif color_type == 'yellow': box_color = (0, 255, 255)
            else: box_color = (128, 128, 128)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
            
            if self.config.DEBUG_MODE and metrics:
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


# =============================================================================
# UTILITY FUNCTIONS (Không dùng nữa nhưng để lại)
# =============================================================================
def convert_roi_to_pixels(roi_percent: Tuple, w_full: int, h_full: int) -> Tuple:
    return (0,0,0,0)
def is_bbox_in_roi(bbox: Tuple, roi_params: Tuple, overlap_threshold: float = 0.5) -> bool:
    return True


# =============================================================================
# TRACKBAR CONTROL (Sử dụng MSER)
# =============================================================================
class TrackbarController:
    # (Không thay đổi so với bản MSER trước)
    def __init__(self, config: TrafficSignConfig):
        self.config = config
        self.window_name = "Parameter Controls (MSER)"
        
    def create_trackbars(self):
        cv2.namedWindow(self.window_name)
        cv2.resizeWindow(self.window_name, 500, 500)
        
        cv2.createTrackbar('MSER Delta', self.window_name, 
                          self.config.MSER_DELTA, 20, lambda x: None)
        cv2.createTrackbar('MSER Min Area', self.window_name, 
                          self.config.MSER_MIN_AREA, 1000, lambda x: None)
        cv2.createTrackbar('MSER Max Area', self.window_name, 
                          self.config.MSER_MAX_AREA, 30000, lambda x: None)
        cv2.createTrackbar('MSER Max Var x100', self.window_name, 
                          int(self.config.MSER_MAX_VARIATION * 100), 100, lambda x: None)
        cv2.createTrackbar('MSER Min Div x100', self.window_name, 
                          int(self.config.MSER_MIN_DIVERSITY * 100), 100, lambda x: None)
        cv2.createTrackbar('Circle Circ x100', self.window_name, 
                            int(self.config.SHAPE_PARAMS['circle']['small_circularity'] * 100), 
                            100, lambda x: None)
        cv2.createTrackbar('Triangle Sol x100', self.window_name, 
                            int(self.config.SHAPE_PARAMS['triangle']['min_solidity'] * 100), 
                            100, lambda x: None)
    
    def update_params(self, verbose=False):
        self.config.MSER_DELTA = max(1, cv2.getTrackbarPos('MSER Delta', self.window_name))
        self.config.MSER_MIN_AREA = max(1, cv2.getTrackbarPos('MSER Min Area', self.window_name))
        self.config.MSER_MAX_AREA = max(1, cv2.getTrackbarPos('MSER Max Area', self.window_name))
        self.config.MSER_MAX_VARIATION = cv2.getTrackbarPos('MSER Max Var x100', self.window_name) / 100.0
        self.config.MSER_MIN_DIVERSITY = cv2.getTrackbarPos('MSER Min Div x100', self.window_name) / 100.0
        circ = cv2.getTrackbarPos('Circle Circ x100', self.window_name) / 100.0
        self.config.SHAPE_PARAMS['circle']['small_circularity'] = circ
        self.config.SHAPE_PARAMS['circle']['large_circularity'] = circ
        sol = cv2.getTrackbarPos('Triangle Sol x100', self.window_name) / 100.0
        self.config.SHAPE_PARAMS['triangle']['min_solidity'] = sol
        return True
    
    def get_mask_visibility(self):
        return {}


# =============================================================================
# MAIN FUNCTION (REAL-TIME MSER)
# =============================================================================
def main():
    print("=" * 70)
    print("  TRAFFIC SIGN DETECTION - REAL-TIME PARAMETER TUNING (MSER Dual Channel)")
    print("=" * 70)
    
    try:
        config = TrafficSignConfig()
        visualizer = Visualizer(config)
        trackbar_ctrl = TrackbarController(config)
        
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
        print("  'q' or ESC: Quit")
        print("  Use trackbars to tune MSER (chạy trên Cr, Cb) and SHAPE parameters")
        
        trackbar_ctrl.create_trackbars()
        
        cv2.namedWindow('Output Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Output Video', w_display, h_display)
        cv2.moveWindow('Output Video', 50, 50)
        
        # (MỚI) Đổi tên cửa sổ mask
        cv2.namedWindow('MSER Mask (Cr+Cb)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MSER Mask (Cr+Cb)', w_display, h_display)
        cv2.moveWindow('MSER Mask (Cr+Cb)', 50, 50 + h_display + 30)
        
        cv2.waitKey(100)
        
        detector = TrafficSignDetector(config)
        frame_count = 0
        paused = False
        verbose_mode = False
        
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        
        print("\n" + "=" * 70)
        print("STARTING REAL-TIME PROCESSING (MSER Dual Channel)...")
        print("=" * 70 + "\n")
        
        current_frame = None
        current_frame_resized = None
        
        while True:
            params_changed = trackbar_ctrl.update_params(verbose=verbose_mode)
            if params_changed:
                detector = TrafficSignDetector(config) # Tạo lại detector với params mới
            
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
                detections, masks = detector.process_frame(current_frame_resized, full_frame_dims)
                
                scale_factor = config.DISPLAY_SCALE / config.PROCESS_SCALE
                scaled_detections = []
                for bbox, color, metrics in detections:
                    x, y, w, h = bbox
                    scaled_bbox = (
                        int(x * scale_factor), int(y * scale_factor),
                        int(w * scale_factor), int(h * scale_factor)
                    )
                    scaled_detections.append((scaled_bbox, color, metrics))
                
                frame_display_sized = cv2.resize(current_frame, (w_display, h_display))
                
                frame_output = visualizer.draw_all(frame_display_sized, frame_count, scaled_detections, current_fps)
                cv2.imshow('Output Video', frame_output)
                
                # Hiển thị MSER mask (kết hợp)
                mser_mask_display = cv2.resize(masks['mser_dual'], (w_display, h_display))
                cv2.imshow('MSER Mask (Cr+Cb)', mser_mask_display)
            
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