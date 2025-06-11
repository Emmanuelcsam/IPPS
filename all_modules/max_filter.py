import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Maximum filtering (dilation)
kernel_size = 7
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
max_filtered = cv2.dilate(img, kernel)


# Display result
cv2.imshow('Maximum Filtered', max_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()