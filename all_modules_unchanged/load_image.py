import cv2

# Load and display image
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)
print(f"Image shape: {img.shape}")
print(f"Data type: {img.dtype}")