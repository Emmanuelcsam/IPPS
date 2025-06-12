import cv2
import numpy as np

def detect_circles(image_path, dp=1.0, min_dist=100, param1=50, param2=30, min_r=0, max_r=0):
    """Detect circles using Hough Transform."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r
    )
    
    if circles is not None:
        circles = np.around(circles[0]).astype(np.uint16)
        for i, (x, y, r) in enumerate(circles):
            print(f"Circle {i}: Center=({x},{y}), Radius={r}")
        return circles
    
    print("No circles detected")
    return []

def draw_circles(image_path, circles):
    """Draw detected circles on image."""
    img = cv2.imread(image_path)
    for x, y, r in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    return img

if __name__ == "__main__":

    circles = detect_circles(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")
    if len(circles) > 0:
        result = draw_circles(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", circles)
        cv2.imwrite("circles_detected.jpg", result)