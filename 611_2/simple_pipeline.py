import cv2
import numpy as np
from pathlib import Path

def load_image(image_path):
    """Load image from file path"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image

def analyze_fiber_optic(image_path):
    """Minimal pipeline for fiber optic analysis."""
    try:
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
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
            if len(pixels) > 0:
                print(f"Core stats: mean={np.mean(pixels):.1f}, std={np.std(pixels):.1f}")
            else:
                print("Core stats: No pixels found")
        
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
            if len(pixels) > 0:
                print(f"Cladding stats: mean={np.mean(pixels):.1f}, std={np.std(pixels):.1f}")
            else:
                print("Cladding stats: No pixels found")
        
        # Draw results
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, (x, y, r) in enumerate(circles[:2]):
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.circle(result, (x, y), r, color, 2)
        
        # Save result with error handling
        output_path = "fiber_analysis_result.jpg"
        success = cv2.imwrite(output_path, result)
        if success:
            print(f"Result saved as {output_path}")
        else:
            print(f"Failed to save result to {output_path}")
        
        # Display results
        cv2.imshow("Fiber Analysis Result", result)
        print("Press any key to close the result window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in fiber analysis: {e}")

def main():
    """Main function with interactive image selection"""
    print("Fiber Optic Analysis Pipeline")
    print("=" * 30)
    
    # Default image path
    default_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
    
    print(f"\nDefault image: {default_path}")
    choice = input("Use default image? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes']:
        image_path = default_path
    else:
        image_path = input("Enter path to your fiber optic image: ").strip()
        # Remove quotes if present
        image_path = image_path.strip('"').strip("'")
    
    # Check if file exists before processing
    if not Path(image_path).exists():
        print(f"Error: File not found - {image_path}")
        return
    
    print(f"\nProcessing image: {image_path}")
    print("-" * 50)
    
    # Run analysis
    analyze_fiber_optic(image_path)
    
    # Test load_image function
    print("\nTesting load_image function...")
    try:
        image = load_image(image_path)
        print(f"Successfully loaded image with shape {image.shape}")
        
        # Display original image
        print("Displaying original image...")
        cv2.imshow("Original Image", image)
        print("Press any key to close the original image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Load image error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()