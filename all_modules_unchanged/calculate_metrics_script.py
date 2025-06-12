import cv2
import numpy as np

# Load ground truth and prediction images
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
ground_truth = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# For demo: create a prediction by thresholding
# In practice, load your actual prediction image
_, prediction = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)

# Convert to binary
gt = (ground_truth > 127).astype(np.uint8)
pred = (prediction > 127).astype(np.uint8)

# Calculate metrics
tp = np.sum((gt == 1) & (pred == 1))
fp = np.sum((gt == 0) & (pred == 1))
fn = np.sum((gt == 1) & (pred == 0))
tn = np.sum((gt == 0) & (pred == 0))

recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print(f"Recall: {recall * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives: {tn}")