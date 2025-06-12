"""Load and display an image"""
import cv2

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Failed to load image: {image_path}")
else:
    print(f"Image loaded: {image.shape}")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()