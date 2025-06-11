import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Add elliptical defect
h, w = image.shape
center_x = int(w * 0.625)
center_y = int(h * 0.45)
cv2.ellipse(image, (center_x, center_y), (15, 8), 45, 0, 360, 180, -1)

# Display result
cv2.imshow('Ellipse Defect Added', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('ellipse_defect_result.jpg', image)