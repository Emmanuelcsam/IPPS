import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=min(gray.shape) // 2
)

# Draw detected circles
output = image.copy()
if circles is not None:
    circles = np.around(circles).astype(np.uint16)[0]
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
    print(f"Detected {len(circles)} circles")
else:
    print("No circles detected")

# Display result
cv2.imshow('Hough Circles', output)
cv2.waitKey(0)
cv2.destroyAllWindows()