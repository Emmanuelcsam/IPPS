import cv2
import numpy as np

# Display image information
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path)

if img is not None:
    # Get image properties
    height, width, channels = img.shape
    size = img.size
    dtype = img.dtype
    
    # Calculate basic statistics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    
    # Create info display
    info_img = np.zeros((300, 400, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(info_img, f"Size: {width}x{height}", (10, 30), font, 0.7, (255, 255, 255), 1)
    cv2.putText(info_img, f"Channels: {channels}", (10, 60), font, 0.7, (255, 255, 255), 1)
    cv2.putText(info_img, f"Type: {dtype}", (10, 90), font, 0.7, (255, 255, 255), 1)
    cv2.putText(info_img, f"Mean: {mean_val:.2f}", (10, 120), font, 0.7, (255, 255, 255), 1)
    cv2.putText(info_img, f"Std: {std_val:.2f}", (10, 150), font, 0.7, (255, 255, 255), 1)
    cv2.putText(info_img, f"Min: {min_val} at {min_loc}", (10, 180), font, 0.7, (255, 255, 255), 1)
    cv2.putText(info_img, f"Max: {max_val} at {max_loc}", (10, 210), font, 0.7, (255, 255, 255), 1)
    
    cv2.imshow('Image', img)
    cv2.imshow('Image Info', info_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Failed to load image from {image_path}")