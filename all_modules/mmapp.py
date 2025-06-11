import cv2
import numpy as np

# Load original and heatmap
original = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")
heatmap = cv2.imread("black_to_red_heatmap.png")

# Convert grayscale to BGR if needed
if len(original.shape) == 2:
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

# Resize heatmap if needed
if original.shape[:2] != heatmap.shape[:2]:
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

# Create overlay
opacity = 0.7
overlay = cv2.addWeighted(original, 1 - opacity, heatmap, opacity, 0)

cv2.imwrite("overlay_result.png", overlay)