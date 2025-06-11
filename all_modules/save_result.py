import cv2
import numpy as np

# Load image and detect circles
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

circles = cv2.HoughCircles(
    blurred, cv2.HOUGH_GRADIENT,
    dp=1.2, minDist=30, param1=50, param2=30,
    minRadius=5, maxRadius=200
)

# Draw circles and save
if circles is not None:
    circles = np.around(circles[0]).astype(np.uint16)
    for x, y, r in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    
    # Save result
    output_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\circles_detected.jpg"
    cv2.imwrite(output_path, img)
    print(f"Result saved to: {output_path}")
else:
    print("No circles found to save")