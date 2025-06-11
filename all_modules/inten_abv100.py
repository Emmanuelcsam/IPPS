import cv2
import numpy as np

# Read image
img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

# Create mask and apply
mask = gray > 100
result = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8) * 255)

# Display result
cv2.imshow('Filtered', result)
cv2.waitKey(0)
cv2.destroyAllWindows()