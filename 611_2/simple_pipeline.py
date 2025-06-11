import cv2
import numpy as np

def analyze_fiber_optic(image_path):
    """Minimal pipeline for fiber optic analysis."""
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=100
    )
    
    if circles is None:
        print("No circles detected")
        return
    
    circles = np.uint16(np.around(circles[0]))
    circles = sorted(circles, key=lambda x: x[2])  # Sort by radius
    
    # Process core (smallest circle)
    if len(circles) >= 1:
        x, y, r = circles[0]
        print(f"Core: center=({x},{y}), radius={r}")
        
        # Create core mask and analyze
        mask = np.zeros_like(img)
        cv2.circle(mask, (x, y), r, 255, -1)
        core_region = cv2.bitwise_and(img, img, mask=mask)
        
        pixels = core_region[core_region > 0]
        print(f"Core stats: mean={np.mean(pixels):.1f}, std={np.std(pixels):.1f}")
    
    # Process cladding (if second circle exists)
    if len(circles) >= 2:
        x2, y2, r2 = circles[1]
        print(f"Cladding: center=({x2},{y2}), radius={r2}")
        
        # Create annular mask
        outer_mask = np.zeros_like(img)
        cv2.circle(outer_mask, (x2, y2), r2, 255, -1)
        inner_mask = np.zeros_like(img)
        cv2.circle(inner_mask, (x, y), r, 255, -1)
        clad_mask = cv2.subtract(outer_mask, inner_mask)
        
        clad_region = cv2.bitwise_and(img, img, mask=clad_mask)
        pixels = clad_region[clad_region > 0]
        print(f"Cladding stats: mean={np.mean(pixels):.1f}, std={np.std(pixels):.1f}")
    
    # Draw results
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, (x, y, r) in enumerate(circles[:2]):
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        cv2.circle(result, (x, y), r, color, 2)
    
    cv2.imwrite("fiber_analysis_result.jpg", result)
    print("Result saved as fiber_analysis_result.jpg")

if __name__ == "__main__":
    analyze_fiber_optic("fiber_optic_image.jpg")