import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Detect circles
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=100
    )
    
    # Convert to color for drawing
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        circles = sorted(circles, key=lambda x: x[2])  # Sort by radius
        
        # Draw circles with different colors
        for i, (x, y, r) in enumerate(circles[:2]):  # Draw first two circles
            if i == 0:
                # Core - green
                cv2.circle(result, (x, y), r, (0, 255, 0), 2)
                cv2.putText(result, "Core", (x-20, y-r-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Cladding - blue
                cv2.circle(result, (x, y), r, (255, 0, 0), 2)
                cv2.putText(result, "Cladding", (x-30, y-r-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw center point
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
    
    cv2.imshow("Annotated Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()