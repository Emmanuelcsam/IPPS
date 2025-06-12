import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path)

# Create a sample defect mask (replace with your actual defect mask)
defect_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
# Example: add some defects
cv2.circle(defect_mask, (150, 150), 30, 255, -1)
cv2.rectangle(defect_mask, (200, 200), (250, 250), 255, -1)

# Overlay defects with color and transparency
color = (0, 0, 255)  # Red in BGR
alpha = 0.5

# Convert grayscale to BGR if needed
if len(image.shape) == 2:
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
else:
    result = image.copy()

# Create colored overlay
overlay = result.copy()
mask = defect_mask > 0
overlay[mask] = color

# Blend with original
result = cv2.addWeighted(result, 1-alpha, overlay, alpha, 0)

# Save and display
cv2.imwrite('defects_overlay.jpg', result)
cv2.imshow('Defects Overlay', result)
cv2.waitKey(0)
cv2.destroyAllWindows()