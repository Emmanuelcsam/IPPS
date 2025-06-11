import cv2
import numpy as np

# Load image
img = cv2.imread(r'C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect circles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                          param1=100, param2=30, minRadius=10, maxRadius=100)

# Create transparent overlay
overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

# Copy original image for display
display_img = img.copy()

# Draw hollow blue circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw on overlay (with alpha)
        cv2.circle(overlay, (i[0], i[1]), i[2], (255, 0, 0, 255), 2)
        # Draw on display image
        cv2.circle(display_img, (i[0], i[1]), i[2], (255, 0, 0), 2)

# Save overlay
cv2.imwrite('circle_overlay.png', overlay)

# Display result
cv2.imshow('Detected Circles', display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()