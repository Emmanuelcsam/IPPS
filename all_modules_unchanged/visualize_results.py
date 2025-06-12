import cv2
import numpy as np

# Load images
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
enhanced = cv2.equalizeHist(original)

# Create a simple scratch detection result for demo
_, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Resize all images to same size
h, w = 300, 300
original_resized = cv2.resize(original, (w, h))
enhanced_resized = cv2.resize(enhanced, (w, h))
result_resized = cv2.resize(result, (w, h))

# Create side-by-side visualization
gap = 10
combined = np.ones((h, w * 3 + gap * 2), dtype=np.uint8) * 255
combined[:, :w] = original_resized
combined[:, w + gap:2 * w + gap] = enhanced_resized
combined[:, 2 * w + 2 * gap:] = result_resized

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(combined, 'Original', (10, 20), font, 0.5, 0, 1)
cv2.putText(combined, 'Enhanced', (w + gap + 10, 20), font, 0.5, 0, 1)
cv2.putText(combined, 'Result', (2 * w + 2 * gap + 10, 20), font, 0.5, 0, 1)

# Save and display
cv2.imwrite('visualization.jpg', combined)
cv2.imshow('LEI Results', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
