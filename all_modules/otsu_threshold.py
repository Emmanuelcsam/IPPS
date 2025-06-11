import cv2

# Load image and convert to grayscale
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply OTSU thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Save result
cv2.imwrite("otsu_threshold_output.jpg", thresh)