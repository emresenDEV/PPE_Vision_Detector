import cv2
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image


def draw_detections(image_path: str, detections: List[Dict], output_path: str = None) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    
    Color coding:
    - GREEN: Compliant (has hardhat, confidence > 0.7)
    - YELLOW: Warning/Uncertain (has hardhat but confidence 0.5-0.7, or no hardhat but confidence 0.5-0.7)
    - RED: Violation (no hardhat, confidence > 0.7)
    
    Args:
        image_path: Path to input image
        detections: List of detection dictionaries
        output_path: Optional path to save annotated image
        
    Returns:
        Annotated image as numpy array
    """
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw each detection
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        has_hardhat = detection['has_hardhat']
        
        # Coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Determine color based on confidence and compliance
        if confidence < 0.5:
            # Skip very low confidence detections
            continue
        elif confidence >= 0.7:
            # High confidence detection
            if has_hardhat:
                color = (0, 255, 0)  # Green - compliant
                status = "COMPLIANT"
            else:
                color = (255, 0, 0)  # Red - violation
                status = "VIOLATION"
        else:
            # Medium confidence (0.5-0.7) - uncertain
            color = (255, 255, 0)  # Yellow - warning
            status = "WARNING"
        
        box_color_bgr = (color[2], color[1], color[0])  # Convert to BGR for OpenCV
        
        # Draw bounding box (thicker for high confidence)
        thickness = 3 if confidence >= 0.7 else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color_bgr, thickness)
        
        # Create label
        label = f"{status} ({confidence:.2f})"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_w, label_h = label_size
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), box_color_bgr, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Annotated image saved to: {output_path}")
    
    # Convert back to RGB for display
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_summary_overlay(image: np.ndarray, summary: Dict) -> np.ndarray:
    """
    Add compliance summary overlay to image.
    
    Args:
        image: Input image (RGB numpy array)
        summary: Compliance summary dictionary
        
    Returns:
        Image with overlay
    """
    # Convert to BGR for OpenCV
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Summary text
    total = summary['total_people']
    compliant = summary['compliant']
    violations = summary['violations']
    rate = summary['compliance_rate']
    status = summary['status']
    
    # Create overlay box
    overlay = img.copy()
    height, width = img.shape[:2]
    
    # Box dimensions
    box_height = 120
    box_width = 350
    x1, y1 = 20, height - box_height - 20
    x2, y2 = x1 + box_width, y1 + box_height
    
    # Draw semi-transparent box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    
    # Add text
    texts = [
        f"Status: {status}",
        f"Total People: {total}",
        f"Compliant: {compliant}",
        f"Violations: {violations}",
        f"Compliance Rate: {rate:.1f}%"
    ]
    
    y_offset = y1 + 25
    for text in texts:
        cv2.putText(img, text, (x1 + 10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    
    # Convert back to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    print("Visualization module loaded successfully!")