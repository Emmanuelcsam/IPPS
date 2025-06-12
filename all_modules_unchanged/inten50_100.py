
#Processing Seperation
import cv2
import numpy as np

# Read image and convert to grayscale
img = cv2.imread('/home/jarvis/Documents/GitHub/OpenCV-Practice/samples2/img38.jpg', 0)

# Create binary mask for pixels between 50-100
lowerb = np.array([50], dtype=np.uint8)
upperb = np.array([100], dtype=np.uint8)
mask = cv2.inRange(img, lowerb, upperb)

# Apply mask to original image
result = cv2.bitwise_and(img, img, mask=mask)

# Display result
cv2.imshow('Filtered', result)
cv2.waitKey(0)
cv2.destroyAllWindows()