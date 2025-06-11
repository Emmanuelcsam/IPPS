import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Add line defects (scratches)
h, w = image.shape

# Diagonal scratch
pt1 = (int(w * 0.25), int(h * 0.25))
pt2 = (int(w * 0.75), int(h * 0.75))
cv2.line(image, pt1, pt2, 160, 2)

# Horizontal scratch
pt3 = (int(w * 0.375), h // 2)
pt4 = (int(w * 0.625), h // 2)
cv2.line(image, pt3, pt4, 155, 1)

# Display result
cv2.imshow('Line Defects Added', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('line_defect_result.jpg', image)