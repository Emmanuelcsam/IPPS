import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Add circular defect
h, w = image.shape
defect_x = int(w * 0.375)
defect_y = int(h * 0.375)
cv2.circle(image, (defect_x, defect_y), 10, 200, -1)

# Display result
cv2.imshow('Circle Defect Added', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('circle_defect_result.jpg', image)