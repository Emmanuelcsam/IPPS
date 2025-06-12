import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)

# Convert to color if grayscale
if len(image.shape) == 2:
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
else:
    result = image.copy()

# Get center
h, w = result.shape[:2]
center = (w // 2, h // 2)

# Draw inspection zones
cv2.circle(result, center, 50, (0, 255, 0), 2)    # Core zone (green)
cv2.circle(result, center, 125, (0, 255, 255), 2) # Cladding zone (yellow)
cv2.circle(result, center, 150, (255, 0, 0), 2)   # Adhesive zone (blue)

# Display result
cv2.imshow('Inspection Zones', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('zones_result.jpg', result)