#!/usr/bin/env python3
"""
Fiber Locator Module
Detects fiber cladding and core using Hough circles and intensity analysis
"""
import cv2
import numpy as np
import os
from pathlib import Path


def load_image(image_path):
    """
    Load and validate image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image or None if failed
    """
    if not image_path:
        print("Error: No image path provided")
        return None
    
    # Convert to string if Path object
    image_path = str(image_path)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found")
        return None
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    file_ext = Path(image_path).suffix.lower()
    
    if file_ext not in valid_extensions:
        print(f"Warning: '{file_ext}' may not be a supported image format")
    
    try:
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from '{image_path}'")
            return None
            
        print(f"Successfully loaded image: {image.shape[1]}x{image.shape[0]} pixels")
        return image
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def find_cladding(image):
    """
    Find fiber cladding using HoughCircles
    Returns: (center_x, center_y), radius
    """
    if image is None or image.size == 0:
        return None, None
        
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = image.shape[:2]
    min_radius = int(min(h, w) * 0.1)
    max_radius = int(min(h, w) * 0.45)
    
    # Ensure minimum radius is reasonable
    min_radius = max(min_radius, 10)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 1)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.15),
        param1=70,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if len(circles) > 0:
            x, y, r = circles[0]  # Take the first (strongest) circle
            # Validate circle is within image bounds
            if (x - r >= 0 and x + r < w and 
                y - r >= 0 and y + r < h):
                return (x, y), r
    
    # Fallback: Use contours
    try:
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 5:
                (x, y), radius = cv2.minEnclosingCircle(largest)
                x, y, radius = int(x), int(y), int(radius)
                # Validate circle is within reasonable bounds
                if (radius >= min_radius and radius <= max_radius and
                    x - radius >= 0 and x + radius < w and 
                    y - radius >= 0 and y + radius < h):
                    return (x, y), radius
    except Exception as e:
        print(f"Warning: Contour fallback failed: {e}")
    
    return None, None


def find_core(image, cladding_center, cladding_radius):
    """
    Find fiber core using intensity analysis
    Returns: (center_x, center_y), radius
    """
    if cladding_center is None or cladding_radius is None:
        return None, None
    
    if image is None or image.size == 0:
        return None, None
        
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cx, cy = cladding_center
    h, w = image.shape[:2]
    
    # Validate cladding center is within image bounds
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None, None
    
    # Create mask for core search area (inner 30% of cladding)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    search_radius = max(int(cladding_radius * 0.3), 5)  # Ensure minimum search radius
    cv2.circle(mask, (cx, cy), search_radius, (255,), -1)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 1)
    
    # Intensity-based detection: analyze radial profile
    max_radius = min(search_radius, min(cx, cy, w-cx, h-cy))  # Stay within bounds
    best_radius = 0
    max_gradient = 0
    
    for r in range(3, max_radius, 2):
        try:
            # Create ring mask
            ring_mask = np.zeros_like(mask)
            cv2.circle(ring_mask, (cx, cy), r + 2, (255,), -1)
            cv2.circle(ring_mask, (cx, cy), max(r-1, 0), (0,), -1)
            
            # Calculate mean intensity in ring
            ring_pixels = blurred[ring_mask > 0]
            if len(ring_pixels) > 10:
                inner_mask = np.zeros_like(mask)
                cv2.circle(inner_mask, (cx, cy), r, (255,), -1)
                inner_pixels = blurred[inner_mask > 0]
                
                if len(inner_pixels) > 10:
                    # Gradient between inner and ring
                    gradient = abs(np.mean(inner_pixels) - np.mean(ring_pixels))
                    if gradient > max_gradient:
                        max_gradient = gradient
                        best_radius = r
        except Exception:
            continue
    
    # Validate detected radius
    if best_radius > 3 and best_radius < cladding_radius * 0.15:
        return cladding_center, best_radius
    
    # Fallback: Use typical ratio (9µm core in 125µm cladding)
    fallback_radius = max(int(cladding_radius * 0.072), 3)
    return cladding_center, fallback_radius


def locate_fiber(image):
    """
    Complete fiber localization
    Returns: dict with cladding and core parameters
    """
    if image is None:
        print("Error: Input image is None")
        return None
    
    if image.size == 0:
        print("Error: Input image is empty")
        return None
    
    try:
        cladding_center, cladding_radius = find_cladding(image)
        
        if cladding_center is None:
            print("Warning: Could not locate fiber cladding")
            return None
        
        core_center, core_radius = find_core(image, cladding_center, cladding_radius)
        
        return {
            'cladding_center': cladding_center,
            'cladding_radius': cladding_radius,
            'core_center': core_center if core_center else cladding_center,
            'core_radius': core_radius if core_radius else max(int((cladding_radius or 0) * 0.072), 3)
        }
    except Exception as e:
        print(f"Error in fiber localization: {e}")
        return None


def locate_fiber_from_file(image_path):
    """
    Load image from file and perform fiber localization
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict with fiber parameters or None if failed
    """
    image = load_image(image_path)
    if image is None:
        return None
    
    return locate_fiber(image)


def display_results(image, result, window_name="Fiber Localization"):
    """
    Display fiber localization results
    
    Args:
        image: Original image
        result: Localization result dictionary
        window_name: Window title
    """
    if image is None or result is None:
        print("Cannot display: missing image or results")
        return
    
    # Create visualization
    vis_img = image.copy()
    if len(vis_img.shape) == 2:  # Convert grayscale to BGR
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    
    # Draw cladding circle
    cv2.circle(vis_img, result['cladding_center'], result['cladding_radius'], (0, 255, 0), 2)
    cv2.putText(vis_img, "Cladding", 
                (result['cladding_center'][0] + result['cladding_radius'] + 5, 
                 result['cladding_center'][1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw core circle
    cv2.circle(vis_img, result['core_center'], result['core_radius'], (0, 0, 255), 2)
    cv2.putText(vis_img, "Core", 
                (result['core_center'][0] + result['core_radius'] + 5, 
                 result['core_center'][1] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Resize if image is too large
    h, w = vis_img.shape[:2]
    max_display_size = 800
    if max(h, w) > max_display_size:
        scale = max_display_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        vis_img = cv2.resize(vis_img, (new_w, new_h))
    
    cv2.imshow(window_name, vis_img)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    Main function with interactive image loading
    """
    print("Fiber Locator - Interactive Mode")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Load image from file path")
        print("2. Run test with synthetic image")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter the path to your fiber image: ").strip()
            # Remove quotes if present
            image_path = image_path.strip('"').strip("'")
            
            # Load and analyze image
            image = load_image(image_path)
            if image is not None:
                print("Analyzing fiber...")
                result = locate_fiber(image)
                
                if result:
                    print(f"\nResults:")
                    print(f"Cladding: center={result['cladding_center']}, radius={result['cladding_radius']}")
                    print(f"Core: center={result['core_center']}, radius={result['core_radius']}")
                    
                    # Ask if user wants to display results
                    show = input("Display results? (y/n): ").strip().lower()
                    if show in ['y', 'yes']:
                        display_results(image, result)
                else:
                    print("Failed to locate fiber in the image")
        
        elif choice == '2':
            print("Running test with synthetic image...")
            try:
                # Create test image
                test_img = np.zeros((200, 200), dtype=np.uint8)
                cv2.circle(test_img, (100, 100), 80, (200,), -1)  # Cladding
                cv2.circle(test_img, (100, 100), 8, (100,), -1)   # Core
                
                # Test localization
                result = locate_fiber(test_img)
                if result:
                    print(f"Cladding: center={result['cladding_center']}, radius={result['cladding_radius']}")
                    print(f"Core: center={result['core_center']}, radius={result['core_radius']}")
                    
                    # Display results
                    display_results(test_img, result, "Test - Fiber Localization")
                    print("Test completed successfully")
                else:
                    print("Failed to locate fiber")
            except Exception as e:
                print(f"Test failed with error: {e}")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


# Test function (for when imported as module)
if __name__ == "__main__":
    main()
