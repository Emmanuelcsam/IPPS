import cv2

# Load image and convert to grayscale
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display result
cv2.imshow('Grayscale', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()