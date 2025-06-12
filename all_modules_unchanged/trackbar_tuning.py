import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create window and trackbars
cv2.namedWindow('Result')
cv2.createTrackbar('param1', 'Result', 50, 200, lambda x: None)
cv2.createTrackbar('param2', 'Result', 30, 100, lambda x: None)

print("Adjust trackbars. Press 'q' to quit.")

while True:
    # Get trackbar values
    param1 = cv2.getTrackbarPos('param1', 'Result')
    param2 = cv2.getTrackbarPos('param2', 'Result')
    
    # Detect circles
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=50, param1=param1, param2=param2,
        minRadius=10, maxRadius=150
    )
    
    # Draw circles
    result = img.copy()
    if circles is not None:
        circles = np.around(circles[0]).astype(np.uint16)
        for x, y, r in circles:
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
    
    cv2.imshow('Result', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()