import cv2

# Load image and count total pixels
img = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)
pixel_count = img.size
print(f"Total pixels: {pixel_count}")