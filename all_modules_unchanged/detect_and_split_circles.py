import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect circles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                          param1=50, param2=30, minRadius=10, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    # Sort circles by radius to find inner and outer
    sorted_circles = sorted(circles[0], key=lambda x: x[2])
    
    if len(sorted_circles) >= 2:
        inner = sorted_circles[0]  # Smallest radius
        outer = sorted_circles[-1]  # Largest radius
        
        # Create masks
        inner_mask = np.zeros(img.shape[:2], np.uint8)
        outer_mask = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(inner_mask, (inner[0], inner[1]), inner[2], 255, -1)
        cv2.circle(outer_mask, (outer[0], outer[1]), outer[2], 255, -1)
        
        # Create ring mask
        ring_mask = cv2.subtract(outer_mask, inner_mask)
        
        # Apply masks
        inner_img = cv2.bitwise_and(img, img, mask=inner_mask)
        ring_img = cv2.bitwise_and(img, img, mask=ring_mask)
        
        # Save results
        cv2.imwrite("detected_inner.png", inner_img)
        cv2.imwrite("detected_ring.png", ring_img)
        print(f"Inner circle: center=({inner[0]}, {inner[1]}), radius={inner[2]}")
        print(f"Outer circle: center=({outer[0]}, {outer[1]}), radius={outer[2]}")
    else:
        print(f"Need at least 2 circles, found {len(sorted_circles)}")
else:
    print("No circles detected")