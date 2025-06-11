import cv2
import numpy as np

# Load original and result images
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
result = cv2.imread('do2mr_result.jpg', cv2.IMREAD_GRAYSCALE)

# Create side-by-side display
display = np.hstack([original, result])

# Display the result
cv2.imshow('Original vs DO2MR Result', display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save combined image
cv2.imwrite('comparison.jpg', display)
