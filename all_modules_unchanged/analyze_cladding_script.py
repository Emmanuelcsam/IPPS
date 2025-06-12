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
    
    if circles is None or len(circles[0]) < 2:
        print("Not enough circles detected for cladding analysis")
    else:
        circles = np.uint16(np.around(circles[0]))
        circles = sorted(circles, key=lambda x: x[2])  # Sort by radius
        
        # Get core and cladding circles
        x1, y1, r1 = circles[0]  # Core (inner)
        x2, y2, r2 = circles[1]  # Cladding (outer)
        
        print(f"Core: radius={r1}, Cladding: radius={r2}")
        
        # Create annular mask for cladding
        outer_mask = np.zeros_like(img)
        cv2.circle(outer_mask, (x2, y2), r2, 255, -1)
        inner_mask = np.zeros_like(img)
        cv2.circle(inner_mask, (x1, y1), r1, 255, -1)
        cladding_mask = cv2.subtract(outer_mask, inner_mask)
        
        # Extract cladding region
        cladding_region = cv2.bitwise_and(img, img, mask=cladding_mask)
        
        # Analyze cladding statistics
        pixels = cladding_region[cladding_region > 0]
        if len(pixels) > 0:
            print(f"Cladding statistics:")
            print(f"  Mean intensity: {np.mean(pixels):.1f}")
            print(f"  Std deviation: {np.std(pixels):.1f}")
            print(f"  Min: {np.min(pixels)}, Max: {np.max(pixels)}")
        
        # Display cladding region
        cv2.imshow("Cladding Region", cladding_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()