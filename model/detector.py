from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple


class PPEDetector:
    """
    PPE detection using YOLOv8 pre-trained model.
    Detects hard hats and classifies as violation or compliant.
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the PPE detector.
        
        Args:
            model_name: YOLOv8 model variant (yolov8n, yolov8s, yolov8m)
            confidence_threshold: Minimum confidence for detections (0.0 to 1.0)
        """
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)  # Auto-downloads if not present
        self.confidence_threshold = confidence_threshold
        
        # Classes we're interested in (from COCO dataset)
        # Class 0 = person
        # We'll use person detection + location to infer PPE compliance
        self.target_classes = [0]  # person class
        
        print(f"Model loaded successfully. Confidence threshold: {confidence_threshold}")
    
    def detect(self, image_path: str) -> List[Dict]:
        """
        Run PPE detection on an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detection dictionaries with bbox, class, confidence
        """
        print(f"Running detection on: {image_path}")
        
        # Run inference
        results = self.model(image_path)
        
        # Process results
        detections = []
        
        for result in results:
            # Get bounding boxes
            boxes = result.boxes
            
            for box in boxes:
                # Extract box data
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # Filter by confidence and target classes
                if confidence >= self.confidence_threshold and class_id in self.target_classes:
                    # For hackathon: Assume anyone visible without helmet = violation
                    # In production: Would use custom trained model for actual helmet detection
                    detection = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': result.names[class_id],
                        'has_hardhat': self._check_hardhat(bbox, image_path),  # Simplified check
                    }
                    
                    detections.append(detection)
        
        print(f"Found {len(detections)} detections")
        return detections
    
    def _check_hardhat(self, bbox: List[float], image_path: str) -> bool:
        """
        Check for hardhat in the detected person's bounding box.

        NOTE: For hackathon MVP, this uses a simplified approach.
        For production, we would train a custom YOLOv8 model on the
        Roboflow Hardhat Detection dataset:
        https://universe.roboflow.com/michael-8jeqe/hardhat-detection-iukt9

        This dataset contains 2,800+ images with hardhat/no-hardhat annotations.
        Training would take 1-2 hours and improve accuracy to 93%+ mAP.

        Citation:
        @misc{ hardhat-detection-iukt9_dataset,
            title = { Hardhat Detection Dataset },
            author = { Michael },
            url = { https://universe.roboflow.com/michael-8jeqe/hardhat-detection-iukt9 },
            publisher = { Roboflow },
            year = { 2025 }
        }

        Current approach (DEMO/MVP ONLY):
        Since COCO dataset does not contain hardhat classes, we use a conservative
        color-based heuristic to detect bright colored objects (yellow, orange, red)
        in the head region which are typical hardhat colors.

        IMPORTANT: This defaults to FALSE (no hardhat) for safety.
        Only returns TRUE if strong evidence of hardhat is found.

        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_path: Path to image

        Returns:
            True if hardhat detected, False otherwise
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            x1, y1, x2, y2 = map(int, bbox)

            # Extract head region (top 25% of person bbox)
            person_height = y2 - y1
            head_region_height = int(person_height * 0.25)
            head_y2 = y1 + head_region_height

            # Ensure coordinates are within image bounds
            head_region = img[max(0, y1):min(img.shape[0], head_y2),
                             max(0, x1):min(img.shape[1], x2)]

            if head_region.size == 0:
                return False

            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

            # Define HSV ranges for common hardhat colors
            # Yellow hardhats (most common)
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])

            # Orange hardhats
            orange_lower = np.array([10, 100, 100])
            orange_upper = np.array([20, 255, 255])

            # Red hardhats (two ranges due to HSV wrap-around)
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 100, 100])
            red_upper2 = np.array([180, 255, 255])

            # White/light hardhats
            white_lower = np.array([0, 0, 200])
            white_upper = np.array([180, 30, 255])

            # Create masks for each color
            mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            mask_white = cv2.inRange(hsv, white_lower, white_upper)

            # Combine all masks
            combined_mask = mask_yellow | mask_orange | mask_red1 | mask_red2 | mask_white

            # Calculate percentage of head region with hardhat colors
            hardhat_pixels = np.sum(combined_mask > 0)
            total_pixels = head_region.shape[0] * head_region.shape[1]
            hardhat_percentage = (hardhat_pixels / total_pixels) * 100

            # Require at least 15% of head region to be hardhat-colored
            # This is conservative - reduces false positives
            if hardhat_percentage >= 15:
                print(f"Hardhat detected: {hardhat_percentage:.1f}% of head region")
                return True
            else:
                print(f"No hardhat: only {hardhat_percentage:.1f}% hardhat-colored pixels")
                return False

        except Exception as e:
            print(f"Error checking hardhat: {e}")
            # Default to False (violation) if check fails - safer for safety application
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