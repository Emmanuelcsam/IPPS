import cv2
import numpy as np

# Load image and convert to grayscale
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Grayscale', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save grayscale
cv2.imwrite('grayscale.png', gray)