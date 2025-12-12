from import ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Tuple, Dict

class PPEDetector:
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the PPE Detector with a YOLOv8 model.
        
        Args: 
        model_name: YOLOv8 (nano)
        confidence_threshold: min confidence for detections (0.0-1.0)
        """
        print(f"Loading YOLOv8 model variant (YOLOv8n): {model_name}")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        self.target_classes - [0]
        print(f"Model loaded successfully. Confidence threshold set to {confidence_threshold}")
        
        def detect(self, impage_path: str) -> List[Dict]:
            """
            Perform PPE detection on the input image.
            
            Args:
            image_path: Path to the input image file.
            
            Returns:
            List of detections with bounding boxes, class labels, and confidence scores.
            """
            print(f"Running detection on image: {image_path}")
            #Run inference
            results = self.model(image_path)
            #Process reults
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    
                    if confidence >= self.confidence_threshold and class_id in self.target_classes:
                        detection = {
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': result.names[class_id],
                            'has_hardhat': self._check_hardhat(bbox, image_path),
                        }
                        detections.append(detection)
                        
            print(f"Found {len(detections)} detections")
            return detections

def _check_hardhat(self, bbox: List[float], image_path: str) -> bool:
    try:
        # Load image
        img = cv2.imread(image_path)
        x1, y1, x2, y2 = map(int, bbox)
        
        #Head region
        person_height = y2 - y1
        head_region_height = int(person_height * 0.3)
        head_y2 = y1 + head_region_height
        
        ##Ensure coordinates are within image bounds
        head_region = img[max(0, y1):min(img.shape[0],head_y2), max(0, x1):min(img.shape[1], x2)]
        if head_region.size == 0:
            return False
        
        head_results = self.model(head_region, verbose=False)
        for result in head_results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id].lower()
                
                if confidence > 0.4 and class_id in [25, 26, 27]:
                    return True
                box_area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                head_area = head_region.shape[0] * head_region.shape[1]
                if box_area / head_area > 0.3:  # Object covers 30%+ of head
                    return True
        return False
    except Exception as e:
        print(f"Error checking hardhat: {e}")
        return False

def classify_compliance(self, detections: List[Dict]) -> Dict:
        """
        Classify overall image compliance based on detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Compliance summary dictionary
        """
        total_people = len(detections)
        people_with_hardhat = sum(1 for d in detections if d['has_hardhat'])
        violations = total_people - people_with_hardhat
        
        compliance_rate = (people_with_hardhat / total_people * 100) if total_people > 0 else 100
        
        summary = {
            'total_people': total_people,
            'compliant': people_with_hardhat,
            'violations': violations,
            'compliance_rate': compliance_rate,
            'status': 'COMPLIANT' if violations == 0 else 'VIOLATION DETECTED'
        }
        
        return summary


def test_detector():
    """Test function for detector module"""
    detector = PPEDetector(confidence_threshold=0.5)
    
    # Test with placeholder
    print("PPEDetector initialized successfully!")
    print(f"Model confidence threshold: {detector.confidence_threshold}")
    

if __name__ == "__main__":
    test_detector()