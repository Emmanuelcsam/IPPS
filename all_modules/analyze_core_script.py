import cv2
import numpy as np

# Load and prepare image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Detect circles
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=100
    )
    
    if circles is None:
        print("No circles detected")
    else:
        circles = np.uint16(np.around(circles[0]))
        circles = sorted(circles, key=lambda x: x[2])  # Sort by radius
        
        # Extract core (smallest circle)
        x, y, r = circles[0]
        print(f"Core: center=({x},{y}), radius={r}")
        
        # Create mask for core
        mask = np.zeros_like(img)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # Extract core region
        core_region = cv2.bitwise_and(img, img, mask=mask)
        
        # Analyze core statistics
        pixels = core_region[core_region > 0]
        if len(pixels) > 0:
            print(f"Core statistics:")
            print(f"  Mean intensity: {np.mean(pixels):.1f}")
            print(f"  Std deviation: {np.std(pixels):.1f}")
            print(f"  Min: {np.min(pixels)}, Max: {np.max(pixels)}")
        
        # Display core region
        cv2.imshow("Core Region", core_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()