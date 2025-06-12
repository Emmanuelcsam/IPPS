import cv2
import numpy as np

# Load and process image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Simple processing - edge detection
    edges = cv2.Canny(img, 50, 150)
    
    # Save the processed image
    output_path = "processed_fiber_optic.jpg"
    success = cv2.imwrite(output_path, edges)
    
    if success:
        print(f"Image saved successfully as: {output_path}")
        # Also save with full path for clarity
        import os
        full_path = os.path.abspath(output_path)
        print(f"Full path: {full_path}")
    else:
        print(f"Failed to save image to {output_path}")