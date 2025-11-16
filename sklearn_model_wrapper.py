"""
SKLEARN MODEL WRAPPER
Allows using sklearn models (.pkl) in hybrid detection pipeline
Compatible với traffic_sign_hybrid_fast.py
"""

import cv2
import numpy as np
import pickle
from typing import Tuple, Optional, List


# =============================================================================
# COLOR HOG EXTRACTOR (Must match training)
# =============================================================================
class ColorHOGExtractor:
    """Extract Color HOG features"""
    
    def __init__(self, win_size=(64, 64), block_size=(16, 16),
                 block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        self.win_size = win_size
        self.hog = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size, nbins
        )
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract Color HOG"""
        if image.shape[:2] != self.win_size:
            image = cv2.resize(image, self.win_size)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        b, g, r = cv2.split(image)
        
        features_b = self.hog.compute(b).flatten()
        features_g = self.hog.compute(g).flatten()
        features_r = self.hog.compute(r).flatten()
        
        return np.hstack((features_b, features_g, features_r)).astype(np.float32)


# =============================================================================
# SKLEARN DETECTOR WRAPPER
# =============================================================================
class SKLearnDetectorWrapper:
    """Wrapper for sklearn detector model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.hog_extractor = ColorHOGExtractor()
        self.load_model()
    
    def load_model(self):
        """Load sklearn model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Loaded Detector (sklearn): {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading detector: {e}")
            raise
    
    def verify(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Verify a detection"""
        x, y, w, h = bbox
        
        # Bounds check
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            return False
        
        if w <= 0 or h <= 0:
            return False
        
        # Crop region
        region = image[y:y+h, x:x+w]
        
        # Extract features
        features = self.hog_extractor.extract(region)
        features = features.reshape(1, -1)
        
        # Predict (sklearn model)
        label = self.model.predict(features)[0]
        
        return label == 1  # Positive class
    
    def verify_batch(self, image: np.ndarray, bboxes: List[Tuple]) -> List[Tuple]:
        """Verify multiple detections"""
        verified = []
        for bbox in bboxes:
            if self.verify(image, bbox):
                verified.append(bbox)
        return verified


# =============================================================================
# SKLEARN RECOGNIZER WRAPPER
# =============================================================================
class SKLearnRecognizerWrapper:
    """Wrapper for sklearn recognizer model"""
    
    def __init__(self, model_path: str, label_map: dict):
        self.model_path = model_path
        self.label_map = label_map
        self.model = None
        self.hog_extractor = ColorHOGExtractor()
        self.load_model()
    
    def load_model(self):
        """Load sklearn model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Loaded Recognizer (sklearn): {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading recognizer: {e}")
            raise
    
    def recognize(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Tuple]:
        """Recognize a single sign"""
        x, y, w, h = bbox
        
        # Bounds check
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            return None
        
        if w <= 0 or h <= 0:
            return None
        
        # Crop region
        region = image[y:y+h, x:x+w]
        
        # Extract features
        features = self.hog_extractor.extract(region)
        features = features.reshape(1, -1)
        
        # Predict
        label = int(self.model.predict(features)[0])
        sign_name = self.label_map.get(label, "unknown")
        
        return (bbox, sign_name, label)
    
    def recognize_batch(self, image: np.ndarray, bboxes: List[Tuple]) -> List[Tuple]:
        """Recognize multiple signs"""
        results = []
        for bbox in bboxes:
            result = self.recognize(image, bbox)
            if result:
                results.append(result)
        return results


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    """Test sklearn models"""
    
    print("\n" + "=" * 70)
    print("🧪 TESTING SKLEARN MODELS")
    print("=" * 70)
    
    # Load models
    detector = SKLearnDetectorWrapper("svm_detector_minibatch_v1.pkl")
    
    recognizer = SKLearnRecognizerWrapper(
        "svm_recognizer_minibatch_v1.pkl",
        label_map={
            0: "cam_di_nguoc_chieu",
            1: "cam_queo_trai",
            2: "cam_do_xe",
            3: "cam_dung_do_xe",
            4: "huong_ben_phai",
            5: "canh_bao_di_cham",
            6: "canh_bao_nguoi_qua_duong",
            7: "canh_bao_duong_gap_khuc"
        }
    )
    
    print("\n✅ Models loaded successfully!")
    print("\nYou can now use these wrappers in traffic_sign_hybrid_fast.py")
    print("\nJust replace:")
    print("   detector_svm = SVMDetector(...)")
    print("With:")
    print("   detector_svm = SKLearnDetectorWrapper('svm_detector_minibatch_v1.pkl')")
    print("\nAnd:")
    print("   recognizer_svm = SVMRecognizer(...)")
    print("With:")
    print("   recognizer_svm = SKLearnRecognizerWrapper('svm_recognizer_minibatch_v1.pkl', LABEL_MAP)")