import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Add rectangular defect
h, w = image.shape
pt1 = (int(w * 0.45), int(h * 0.625))
pt2 = (int(w * 0.475), int(h * 0.65))
cv2.rectangle(image, pt1, pt2, 190, -1)

# Display result
cv2.imshow('Rectangle Defect Added', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('rectangle_defect_result.jpg', image)