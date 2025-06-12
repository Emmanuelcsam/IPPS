import cv2

# Load and convert image to grayscale
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Save result
cv2.imwrite("grayscale_output.jpg", gray)