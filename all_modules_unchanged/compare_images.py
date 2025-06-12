import cv2
import numpy as np

# Compare two images side by side
image_path1 = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image_path2 = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"  # Change to second image path

img1 = cv2.imread(image_path1)
img2 = cv2.imread(image_path2)

if img1 is not None and img2 is not None:
    # Resize images to same height if needed
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 != h2:
        # Scale to same height
        scale = h1 / h2
        img2 = cv2.resize(img2, (int(w2 * scale), h1))
    
    # Concatenate horizontally
    combined = np.hstack((img1, img2))
    
    cv2.imshow('Comparison', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load one or both images")