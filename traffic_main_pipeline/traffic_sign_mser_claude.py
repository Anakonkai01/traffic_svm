"""
TRAFFIC SIGN DETECTION - MSER + HOG GRAYSCALE + COLOR VERIFICATION
Pipeline "Linh hoạt" - Tiết kiệm RAM, Độ chính xác cao

Giai đoạn 1: MSER + Lọc Hình dạng (Tìm ứng viên)
Giai đoạn 2: HOG Xám + SVM Detector (Loại bỏ false positives)
Giai đoạn 2.5: Kiểm tra Màu (Loại bỏ màu không hợp lệ)
Giai đoạn 3: HOG Xám + SVM Recognizer (Nhận dạng)
"""

import cv2
import numpy as np
import time
import gc
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import queue


DRIVE_PATH = "../"

# Models sử dụng HOG Xám (grayscale)
MODEL_DETECTOR_PATH = DRIVE_PATH + "models/svm_sign_detector_v4.xml"
MODEL_RECOGNIZER_PATH = DRIVE_PATH + "models/svm_sign_recognizer_v3.xml"


# =============================================================================
# CONFIGURATION
# =============================================================================
class MSERConfig:
    """Configuration for MSER + Grayscale HOG pipeline"""
    
    def __init__(self):
        # Video Paths
        self.INPUT_VIDEO_PATH = DRIVE_PATH + "videos/video1.mp4"
        self.OUTPUT_VIDEO_PATH = DRIVE_PATH + "videos/video1_mser_claude_output.mp4"
        
        # MSER Parameters (Giai đoạn 1 - Tìm ứng viên)
        self.MSER_DELTA = 5
        self.MSER_MIN_AREA = 100
        self.MSER_MAX_AREA = 15000
        self.MSER_MAX_VARIATION = 0.25
        self.MSER_MIN_DIVERSITY = 0.2
        
        # Shape Filtering (Giai đoạn 1)
        self.MIN_CIRCULARITY = 0.5  # Cho hình tròn
        self.MIN_SOLIDITY = 0.5  # Cho tam giác
        self.MIN_ASPECT_RATIO = 0.7  # Tỷ lệ w/h
        self.MAX_ASPECT_RATIO = 1.3
        
        # HOG Parameters (Grayscale - tiết kiệm RAM!)
        self.HOG_WIN_SIZE = (64, 64)
        self.HOG_BLOCK_SIZE = (16, 16)
        self.HOG_BLOCK_STRIDE = (8, 8)
        self.HOG_CELL_SIZE = (8, 8)
        self.HOG_NBINS = 9
        
        # Color Verification (Giai đoạn 2.5 - MẤU CHỐT!)
        self.VALID_COLORS = {
            'red': {
                'hsv_range_1': (np.array([0, 70, 50]), np.array([10, 255, 255])),
                'hsv_range_2': (np.array([110, 30, 0]), np.array([179, 255, 255]))
            },
            'blue': {
                'hsv_range': (np.array([100, 80, 70]), np.array([145, 255, 230]))
            },
            'yellow': {
                'hsv_range': (np.array([8, 100, 90]), np.array([30, 255, 255]))
            }
        }
        self.COLOR_THRESHOLD = 0.01  # 1% pixels phải có màu hợp lệ
        
        # Processing Settings
        self.NUM_WORKERS = max(1, cpu_count() - 1)
        self.BATCH_SIZE = 16
        self.PREFETCH_FRAMES = 64
        
        # ROI Crop Settings
        self.TOP_CROP_PERCENT = 0.35
        self.RIGHT_CROP_PERCENT = 0.05
        
        # Progress Settings
        self.PROGRESS_UPDATE_INTERVAL = 100
        self.MAX_FRAME_ID = 10000


# =============================================================================
# GRAYSCALE HOG EXTRACTOR
# =============================================================================
class GrayscaleHOGExtractor:
    """Extract HOG features from GRAYSCALE images (1/3 RAM của Color HOG!)"""
    
    def __init__(self, config: MSERConfig):
        self.config = config
        self.hog = cv2.HOGDescriptor(
            config.HOG_WIN_SIZE,
            config.HOG_BLOCK_SIZE,
            config.HOG_BLOCK_STRIDE,
            config.HOG_CELL_SIZE,
            config.HOG_NBINS
        )
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract Grayscale HOG features"""
        # Resize
        if image.shape[:2] != self.config.HOG_WIN_SIZE:
            image = cv2.resize(image, self.config.HOG_WIN_SIZE,
                              interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG
        features = self.hog.compute(image).flatten()
        
        return features.astype(np.float32)


# =============================================================================
# MSER DETECTOR (Giai đoạn 1)
# =============================================================================
class MSERDetector:
    """MSER-based region detector"""
    
    def __init__(self, config: MSERConfig):
        self.config = config
        self.mser = cv2.MSER_create(
            delta=config.MSER_DELTA,
            min_area=config.MSER_MIN_AREA,
            max_area=config.MSER_MAX_AREA,
            max_variation=config.MSER_MAX_VARIATION,
            min_diversity=config.MSER_MIN_DIVERSITY
        )
    
    def detect_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect candidate regions using MSER"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect MSER regions
        regions, _ = self.mser.detectRegions(gray)
        
        # Convert regions to bounding boxes
        bboxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            bboxes.append((x, y, w, h))
        
        return bboxes
    
    def filter_by_shape(self, frame: np.ndarray, 
                       bboxes: List[Tuple]) -> List[Tuple]:
        """Filter bounding boxes by shape properties"""
        filtered = []
        
        for bbox in bboxes:
            x, y, w, h = bbox
            
            # Check aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if (aspect_ratio < self.config.MIN_ASPECT_RATIO or 
                aspect_ratio > self.config.MAX_ASPECT_RATIO):
                continue
            
            # Extract region
            if (x < 0 or y < 0 or x + w > frame.shape[1] or 
                y + h > frame.shape[0]):
                continue
            
            region = frame[y:y+h, x:x+w]
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Threshold
            _, binary = cv2.threshold(gray_region, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Get largest contour
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            
            if area < 100:
                continue
            
            # Check circularity (for circles)
            perimeter = cv2.arcLength(main_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > self.config.MIN_CIRCULARITY:
                    filtered.append(bbox)
                    continue
            
            # Check solidity (for triangles)
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity > self.config.MIN_SOLIDITY:
                    filtered.append(bbox)
        
        return filtered


# =============================================================================
# COLOR VERIFIER (Giai đoạn 2.5 - MẤU CHỐT!)
# =============================================================================
class ColorVerifier:
    """Verify if region has valid traffic sign colors"""
    
    def __init__(self, config: MSERConfig):
        self.config = config
    
    def verify_color(self, frame: np.ndarray, bbox: Tuple) -> bool:
        """
        Kiểm tra xem vùng này có màu hợp lệ không
        Return True nếu có ít nhất COLOR_THRESHOLD % pixels là Đỏ/Xanh/Vàng
        """
        x, y, w, h = bbox
        
        # Extract region
        if (x < 0 or y < 0 or x + w > frame.shape[1] or 
            y + h > frame.shape[0]):
            return False
        
        region = frame[y:y+h, x:x+w]
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Create masks for valid colors
        valid_mask = np.zeros(region.shape[:2], dtype=np.uint8)
        
        # Red (two ranges because red wraps around in HSV)
        red_params = self.config.VALID_COLORS['red']
        mask1 = cv2.inRange(hsv, red_params['hsv_range_1'][0], 
                           red_params['hsv_range_1'][1])
        mask2 = cv2.inRange(hsv, red_params['hsv_range_2'][0],
                           red_params['hsv_range_2'][1])
        valid_mask = cv2.bitwise_or(valid_mask, cv2.bitwise_or(mask1, mask2))
        
        # Blue
        blue_params = self.config.VALID_COLORS['blue']
        mask_blue = cv2.inRange(hsv, blue_params['hsv_range'][0],
                               blue_params['hsv_range'][1])
        valid_mask = cv2.bitwise_or(valid_mask, mask_blue)
        
        # Yellow
        yellow_params = self.config.VALID_COLORS['yellow']
        mask_yellow = cv2.inRange(hsv, yellow_params['hsv_range'][0],
                                 yellow_params['hsv_range'][1])
        valid_mask = cv2.bitwise_or(valid_mask, mask_yellow)
        
        # Calculate percentage of valid color pixels
        valid_pixels = np.count_nonzero(valid_mask)
        total_pixels = region.shape[0] * region.shape[1]
        
        if total_pixels == 0:
            return False
        
        color_ratio = valid_pixels / total_pixels
        
        return color_ratio >= self.config.COLOR_THRESHOLD


# =============================================================================
# SVM DETECTOR (Giai đoạn 2 - HOG Xám)
# =============================================================================
class SVMDetector:
    """SVM-based detector using Grayscale HOG"""
    
    def __init__(self, model_path: str, config: MSERConfig):
        self.model_path = model_path
        self.config = config
        self.svm = None
        self.hog_extractor = GrayscaleHOGExtractor(config)
        self.load_model()
    
    def load_model(self):
        """Load SVM detector model"""
        try:
            self.svm = cv2.ml.SVM_load(self.model_path)
            print(f"✅ Loaded SVM Detector (Grayscale HOG): {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading SVM Detector: {e}")
            raise
    
    def verify(self, frame: np.ndarray, bbox: Tuple) -> bool:
        """Verify if bbox contains a traffic sign"""
        x, y, w, h = bbox
        
        if (x < 0 or y < 0 or x + w > frame.shape[1] or 
            y + h > frame.shape[0]):
            return False
        
        try:
            # Extract region
            region = frame[y:y+h, x:x+w]
            
            # Extract HOG features
            features = self.hog_extractor.extract(region)
            features = features.reshape(1, -1)
            
            # Predict
            _, result = self.svm.predict(features, 
                                        flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            confidence = -result[0][0]
            
            return confidence > 0
        except:
            return False


# =============================================================================
# SVM RECOGNIZER (Giai đoạn 3 - HOG Xám)
# =============================================================================
class SVMRecognizer:
    """SVM-based recognizer using Grayscale HOG"""
    
    def __init__(self, model_path: str, config: MSERConfig):
        self.model_path = model_path
        self.config = config
        self.svm = None
        self.hog_extractor = GrayscaleHOGExtractor(config)
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
            print(f"✅ Loaded SVM Recognizer (Grayscale HOG): {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading SVM Recognizer: {e}")
            raise
    
    def recognize(self, frame: np.ndarray, bbox: Tuple) -> Optional[Tuple]:
        """Recognize a single sign"""
        x, y, w, h = bbox
        
        if (x < 0 or y < 0 or x + w > frame.shape[1] or 
            y + h > frame.shape[0]):
            return None
        
        try:
            # Extract region
            region = frame[y:y+h, x:x+w]
            
            # Extract features
            features = self.hog_extractor.extract(region)
            features = features.reshape(1, -1)
            
            # Predict
            label = int(self.svm.predict(features)[1][0][0])
            sign_name = self.sign_names.get(label, "unknown")
            
            return (bbox, sign_name, label)
        except:
            return None
    
    def recognize_batch(self, frame: np.ndarray, 
                       bboxes: List[Tuple]) -> List[Tuple]:
        """Recognize multiple signs"""
        results = []
        for bbox in bboxes:
            result = self.recognize(frame, bbox)
            if result:
                results.append(result)
        return results


# =============================================================================
# PARALLEL FRAME READER
# =============================================================================
class ParallelFrameReader:
    """Multi-threaded frame reader with prefetching"""
    
    def __init__(self, video_path: str, total_frames: int, prefetch_size: int = 64):
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
        self.frame_queue.put(None)
    
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
    """Draw detection results"""
    
    def __init__(self, config: MSERConfig):
        self.config = config
        self.colors = {
            "cam_di_nguoc_chieu": (0, 0, 255),
            "cam_queo_trai": (0, 0, 255),
            "cam_do_xe": (0, 0, 255),
            "cam_dung_do_xe": (0, 0, 255),
            "huong_ben_phai": (255, 0, 0),
            "canh_bao_di_cham": (0, 255, 255),
            "canh_bao_nguoi_qua_duong": (0, 255, 255),
            "canh_bao_duong_gap_khuc": (0, 255, 255),
        }
    
    def draw_all(self, frame: np.ndarray, frame_num: int,
                 detections: List[Tuple]) -> np.ndarray:
        """Draw all detections"""
        output = frame.copy()
        
        for detection in detections:
            bbox, sign_name, label = detection
            x, y, w, h = bbox
            
            color = self.colors.get(sign_name, (255, 255, 255))
            
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            label_text = sign_name.replace("_", " ").title()
            cv2.putText(output, label_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(output, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output


# =============================================================================
# WORKER FUNCTION
# =============================================================================
def process_frame_mser_worker(args):
    """
    Worker function - 3 Giai đoạn:
    1. MSER + Shape filtering
    2. SVM Detector (HOG Xám)
    2.5. Color Verification
    3. (Sẽ làm ở main thread)
    """
    frame_num, frame, config = args
    
    # Stage 1: MSER Detection
    mser_detector = MSERDetector(config)
    candidate_bboxes = mser_detector.detect_regions(frame)
    candidate_bboxes = mser_detector.filter_by_shape(frame, candidate_bboxes)
    
    # Stage 2: SVM Detector verification
    svm_detector = SVMDetector(MODEL_DETECTOR_PATH, config)
    verified_bboxes = []
    for bbox in candidate_bboxes:
        if svm_detector.verify(frame, bbox):
            verified_bboxes.append(bbox)
    
    # Stage 2.5: Color Verification
    color_verifier = ColorVerifier(config)
    final_bboxes = []
    for bbox in verified_bboxes:
        if color_verifier.verify_color(frame, bbox):
            final_bboxes.append(bbox)
    
    return (frame_num, final_bboxes)


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main processing pipeline"""
    
    print("\n" + "=" * 70)
    print("🎯 MSER + HOG XÁM + KIỂM TRA MÀU")
    print("=" * 70)
    print("Pipeline 'Linh hoạt':")
    print("  ✅ Giai đoạn 1: MSER + Lọc Hình dạng")
    print("  ✅ Giai đoạn 2: HOG Xám + SVM Detector")
    print("  ✅ Giai đoạn 2.5: Kiểm tra Màu (MẤU CHỐT!)")
    print("  ✅ Giai đoạn 3: HOG Xám + SVM Recognizer")
    print("  ✅ Tiết kiệm RAM: HOG Xám chỉ 1/3 HOG Màu")
    print("=" * 70)
    
    try:
        start_total = time.time()
        
        # Initialize configuration
        config = MSERConfig()
        
        # Load models
        detector_svm = SVMDetector(MODEL_DETECTOR_PATH, config)
        recognizer_svm = SVMRecognizer(MODEL_RECOGNIZER_PATH, config)
        
        # Initialize visualizer
        visualizer = Visualizer(config)
        
        # Open video
        cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {config.INPUT_VIDEO_PATH}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        h_crop = int(h_orig * (1 - config.TOP_CROP_PERCENT))
        w_crop = int(w_orig * (1 - config.RIGHT_CROP_PERCENT))
        
        print(f"\n📹 Video Info:")
        print(f"   Resolution: {w_orig}x{h_orig}")
        print(f"   Processing Region: {w_crop}x{h_crop}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        
        # =================================================================
        # PASS 1: MSER + DETECTOR + COLOR VERIFICATION
        # =================================================================
        print("\n" + "=" * 70)
        print("🔍 PASS 1: MSER + DETECTOR + COLOR VERIFICATION")
        print("=" * 70)
        start_detection = time.time()
        
        frame_reader = ParallelFrameReader(config.INPUT_VIDEO_PATH,
                                          config.MAX_FRAME_ID,
                                          config.PREFETCH_FRAMES)
        frame_reader.start()
        
        all_frame_detections = {}
        frame_batch = []
        frame_count = 0
        
        with Pool(processes=config.NUM_WORKERS) as pool:
            while True:
                item = frame_reader.get_frame()
                if item is None:
                    break
                
                frame_num, frame_full = item
                frame_to_process = frame_full[0:h_crop, 0:w_crop]
                
                frame_batch.append((frame_num, frame_to_process, config))
                
                if len(frame_batch) >= config.BATCH_SIZE:
                    results = pool.map(process_frame_mser_worker, frame_batch)
                    
                    for frame_num, bboxes in results:
                        all_frame_detections[frame_num] = bboxes
                    
                    frame_count += len(frame_batch)
                    frame_batch.clear()
                    
                    if frame_count % config.PROGRESS_UPDATE_INTERVAL == 0:
                        progress = int(frame_count / config.MAX_FRAME_ID * 100)
                        elapsed = time.time() - start_detection
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"   Processed {frame_count}/{config.MAX_FRAME_ID} "
                              f"({progress}%) - {current_fps:.1f} FPS")
                        gc.collect()
            
            if frame_batch:
                results = pool.map(process_frame_mser_worker, frame_batch)
                for frame_num, bboxes in results:
                    all_frame_detections[frame_num] = bboxes
                frame_count += len(frame_batch)
        
        frame_reader.stop()
        
        detection_time = time.time() - start_detection
        total_detections = sum(len(dets) for dets in all_frame_detections.values())
        
        print(f"\n✅ Detection Pass completed in {detection_time:.2f}s")
        print(f"   Average Speed: {frame_count/detection_time:.1f} FPS")
        print(f"   Total detections: {total_detections}")
        
        # =================================================================
        # PASS 2: RECOGNIZE + RENDER
        # =================================================================
        print("\n" + "=" * 70)
        print("🎨 PASS 2: RECOGNIZE + RENDER")
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
            print(f"❌ Cannot create output: {config.OUTPUT_VIDEO_PATH}")
            return
        
        frame_count_render = 0
        total_signs_recognized = 0
        
        while True:
            item = frame_reader.get_frame()
            if item is None:
                break
            
            frame_num, frame = item
            
            bboxes = all_frame_detections.get(frame_num, [])
            
            recognized_detections = []
            if bboxes:
                recognized_detections = recognizer_svm.recognize_batch(frame, bboxes)
                total_signs_recognized += len(recognized_detections)
            
            frame_output = visualizer.draw_all(frame, frame_num, recognized_detections)
            video_writer.write(frame_output)
            
            frame_count_render += 1
            
            if frame_count_render % config.PROGRESS_UPDATE_INTERVAL == 0:
                progress = int(frame_count_render / total_frames * 100)
                elapsed = time.time() - start_render
                current_fps = frame_count_render / elapsed if elapsed > 0 else 0
                print(f"   Rendered {frame_count_render}/{total_frames} "
                      f"({progress}%) - {current_fps:.1f} FPS - "
                      f"{total_signs_recognized} signs")
            
            del frame, frame_output
        
        frame_reader.stop()
        video_writer.release()
        
        render_time = time.time() - start_render
        total_time = time.time() - start_total
        
        print(f"\n✅ Rendering Pass completed in {render_time:.2f}s")
        
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"📁 Output: {config.OUTPUT_VIDEO_PATH}")
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"   Detection: {detection_time:.2f}s")
        print(f"   Rendering: {render_time:.2f}s")
        print(f"   Overall: {frame_count_render/total_time:.1f} FPS")
        print(f"   Total Signs: {total_signs_recognized}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()


if __name__ == "__main__":
    main()