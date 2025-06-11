import cv2

# Load image in grayscale and apply Gaussian blur
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Display result
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()