from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any

class ColorDetector:
    def __init__(self):
        # Define the colour ranges in HSV
        self.colors = {
            'red': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),     # Lower red
                (np.array([160, 100, 100]), np.array([180, 255, 255]))   # Upper red
            ],
            'green': [
                (np.array([40, 40, 40]), np.array([80, 255, 255]))       # Green range
            ],
            'blue': [
                (np.array([100, 50, 50]), np.array([140, 255, 255]))     # Blue range
            ]
        }

    def detect_colors(self, frame: np.ndarray, min_area: int = 1000) -> List[Dict[str, Any]]:
        # Apply median blur to reduce noise
        median_blurred_frame = cv2.medianBlur(frame, 5)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered_frame = cv2.bilateralFilter(median_blurred_frame, d=9, sigmaColor=75, sigmaSpace=75)

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)
        detected_objects = []

        for color_name, color_ranges in self.colors.items():
            for (lower, upper) in color_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                
                # Reduce noise using morphological operations
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Calculate centroid
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = x + w // 2, y + h // 2

                        detected_objects.append({
                            'color': color_name,
                            'bbox': (x, y, w, h),
                            'centroid': (cx, cy),
                            'area': area
                        })

        return detected_objects

# Load trained YOLO model
model = YOLO("MSD_Final.pt")  # Use your own trained weights

# Initialize ColorDetector
color_detector = ColorDetector()

# Start webcam
cap = cv2.VideoCapture(0)

# Color map for drawing bounding boxes
color_map = {
    'red': (0, 0, 255),    # BGR: Red
    'green': (0, 255, 0),  # BGR: Green
    'blue': (255, 0, 0),   # BGR: Blue
    'unknown': (255, 255, 255)  # BGR: White (fallback)
}

def get_box_color(roi: np.ndarray) -> str:
    """
    Determine the color of the ROI using ColorDetector.
    Returns the dominant color or 'unknown' if no color is detected.
    """
    detected_objects = color_detector.detect_colors(roi, min_area=1000)  # Adjusted for smaller ROIs
    
    if not detected_objects:
        return 'unknown'
    
    # Return the color of the largest detected object
    largest_obj = max(detected_objects, key=lambda x: x['area'])
    return largest_obj['color']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect boxes with YOLO
    results = model.predict(source=frame, save=False, conf=0.5)
    
    # Work on a copy of the frame for custom annotations
    annotated_frame = frame.copy()
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:  # Skip empty ROIs
                continue
            
            # Get color using ColorDetector
            color = get_box_color(roi)
            
            # Get class name and confidence
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            
            # Create label with class, color, and confidence
            label = f"{class_name} ({color}) {conf:.2f}"
            
            # Draw bounding box with color-specific color
            draw_color = color_map.get(color, color_map['unknown'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), draw_color, 2)
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), draw_color, -1)
            
            # Draw label text with increased font size and thickness
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)  # Increased fontScale to 1.0 and thickness to 3

    # Show the frame
    cv2.imshow("MSD Project", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()