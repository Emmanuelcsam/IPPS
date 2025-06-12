import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_circular_mask(image_shape, center_x, center_y, radius):
    """
    Create a circular mask using the mathematical approach discussed.
    
    Uses the circle equation: (x - cx)^2 + (y - cy)^2 <= r^2
    
    Args:
        image_shape: (height, width) of the image
        center_x: x-coordinate of circle center
        center_y: y-coordinate of circle center
        radius: radius of the circle
    
    Returns:
        Binary mask where pixels inside the circle are 255, outside are 0
    """
    height, width = image_shape
    
    # Create coordinate grids
    y_indices, x_indices = np.ogrid[:height, :width]
    
    # Calculate distance from each pixel to the center
    # This is the key mathematical formula from the conversation
    distance_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    # Create mask: pixels where distance <= radius are inside the circle
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[distance_from_center <= radius] = 255
    
    return mask


def separate_core_and_cladding(image_path):
    """
    Separate core and cladding regions using circular masking.
    """
    # Step 1: Load and preprocess the image
    print("Loading image...")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return None
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    print(f"Image shape: {gray.shape}")
    print(f"Intensity range: [{gray.min()}, {gray.max()}]")
    
    # Step 2: Detect circles using HoughCircles
    print("\nDetecting circles...")
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,              # Inverse ratio of accumulator resolution
        minDist=50,          # Minimum distance between circles
        param1=50,           # Canny edge detection threshold
        param2=30,           # Circle detection threshold
        minRadius=10,        # Minimum radius
        maxRadius=200        # Maximum radius
    )
    
    if circles is None:
        print("No circles detected. Try adjusting parameters.")
        return None
    
    # Convert to integer coordinates
    circles = np.around(circles).astype(np.uint16)
    
    # Sort circles by radius (smallest first - likely the core)
    circles = circles[0, :]  # Get first row
    circles = sorted(circles, key=lambda x: x[2])  # Sort by radius
    
    print(f"Detected {len(circles)} circles:")
    for i, (x, y, r) in enumerate(circles):
        print(f"  Circle {i}: Center=({x}, {y}), Radius={r}")
    
    # Step 3: Create masks for core and cladding
    if len(circles) < 2:
        print("\nNeed at least 2 circles for core and cladding. Only doing core.")
        core_x, core_y, core_r = circles[0]
        
        # Create core mask
        core_mask = create_circular_mask(gray.shape, core_x, core_y, core_r)
        
        # Apply mask to get core region
        core_only = cv2.bitwise_and(gray, gray, mask=core_mask)
        
        # Display results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(core_mask, cmap='gray')
        plt.title('Core Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(core_only, cmap='gray')
        plt.title('Isolated Core')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'core_image': core_only,
            'cladding_image': None,
            'core_mask': core_mask,
            'cladding_mask': None,
            'core_circle': (core_x, core_y, core_r),
            'cladding_circle': None
        }
    
    # We have at least 2 circles - separate core and cladding
    core_x, core_y, core_r = circles[0]  # Smallest circle (core)
    clad_x, clad_y, clad_r = circles[1]  # Next circle (cladding outer boundary)
    
    print(f"\nCore: Center=({core_x}, {core_y}), Radius={core_r}")
    print(f"Cladding outer: Center=({clad_x}, {clad_y}), Radius={clad_r}")
    
    # Create masks
    core_mask = create_circular_mask(gray.shape, core_x, core_y, core_r)
    cladding_outer_mask = create_circular_mask(gray.shape, clad_x, clad_y, clad_r)
    
    # Cladding mask = outer mask - core mask (annular region)
    cladding_mask = cladding_outer_mask.copy()
    cladding_mask[core_mask == 255] = 0  # Remove core region
    
    # Apply masks to isolate regions
    core_only = cv2.bitwise_and(gray, gray, mask=core_mask)
    cladding_only = cv2.bitwise_and(gray, gray, mask=cladding_mask)
    
    # Calculate intensity statistics for each region
    core_pixels = gray[core_mask == 255].tolist()
    cladding_pixels = gray[cladding_mask == 255].tolist()
    
    if len(core_pixels) > 0:
        print(f"\nCore statistics:")
        print(f"  Mean intensity: {np.mean(core_pixels):.1f}")
        print(f"  Std deviation: {np.std(core_pixels):.1f}")
        print(f"  Min/Max: {np.min(core_pixels)}/{np.max(core_pixels)}")
    else:
        print("\nNo core pixels found!")
    
    if len(cladding_pixels) > 0:
        print(f"\nCladding statistics:")
        print(f"  Mean intensity: {np.mean(cladding_pixels):.1f}")
        print(f"  Std deviation: {np.std(cladding_pixels):.1f}")
        print(f"  Min/Max: {np.min(cladding_pixels)}/{np.max(cladding_pixels)}")
    else:
        print("\nNo cladding pixels found!")
    
    # Visualize results
    try:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # First row - original and detected circles
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title('Original Grayscale')
        axes[0, 0].axis('off')
        
        # Draw circles on image
        img_with_circles = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(img_with_circles, (core_x, core_y), core_r, (0, 255, 0), 2)
        cv2.circle(img_with_circles, (clad_x, clad_y), clad_r, (0, 0, 255), 2)
        cv2.circle(img_with_circles, (core_x, core_y), 2, (255, 0, 0), 3)  # Center point
        
        axes[0, 1].imshow(cv2.cvtColor(img_with_circles, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Detected Circles (Green=Core, Red=Cladding)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(core_mask, cmap='gray')
        axes[0, 2].set_title('Core Mask')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(cladding_mask, cmap='gray')
        axes[0, 3].set_title('Cladding Mask (Annular)')
        axes[0, 3].axis('off')
        
        # Second row - isolated regions and histograms
        axes[1, 0].imshow(core_only, cmap='gray')
        axes[1, 0].set_title('Isolated Core')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cladding_only, cmap='gray')
        axes[1, 1].set_title('Isolated Cladding')
        axes[1, 1].axis('off')
        
        if len(core_pixels) > 0:
            axes[1, 2].hist(core_pixels, bins=50, alpha=0.7, color='green')
            axes[1, 2].set_title('Core Intensity Distribution')
            axes[1, 2].set_xlabel('Intensity')
            axes[1, 2].set_ylabel('Count')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Core Pixels', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Core Intensity Distribution')
        
        if len(cladding_pixels) > 0:
            axes[1, 3].hist(cladding_pixels, bins=50, alpha=0.7, color='red')
            axes[1, 3].set_title('Cladding Intensity Distribution')
            axes[1, 3].set_xlabel('Intensity')
            axes[1, 3].set_ylabel('Count')
        else:
            axes[1, 3].text(0.5, 0.5, 'No Cladding Pixels', ha='center', va='center', transform=axes[1, 3].transAxes)
            axes[1, 3].set_title('Cladding Intensity Distribution')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error displaying plots: {e}")
    
    # Return the separated images and masks
    return {
        'core_image': core_only,
        'cladding_image': cladding_only,
        'core_mask': core_mask,
        'cladding_mask': cladding_mask,
        'core_circle': (core_x, core_y, core_r),
        'cladding_circle': (clad_x, clad_y, clad_r)
    }


def manual_circle_masking_example():
    """
    Example showing the manual mathematical approach discussed in the conversation.
    This demonstrates how to check if a point is inside or outside a circle.
    """
    print("\n=== Manual Circle Masking Example ===")
    
    # Create a small test image
    width, height = 100, 100
    test_img = np.zeros((height, width), dtype=np.uint8)
    
    # Define a circle (as discussed in the conversation)
    center_x, center_y = 50, 50  # Center at (50, 50)
    radius = 30
    
    print(f"Circle: Center=({center_x}, {center_y}), Radius={radius}")
    
    # Manually iterate through pixels (as they discussed)
    for y in range(height):
        for x in range(width):
            # Calculate distance using the equation from the conversation
            # sqrt((x - cx)^2 + (y - cy)^2)
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Check if inside circle (distance <= radius)
            if distance <= radius:
                test_img[y, x] = 255
    
    # Test specific points (as they did in the conversation)
    test_points = [
        (center_x, center_y),  # Center point
        (center_x + radius, center_y),  # On circle edge
        (center_x + radius + 5, center_y),  # Outside circle
        (center_x + 10, center_y + 10)  # Inside circle
    ]
    
    print("\nTesting points:")
    for x, y in test_points:
        if 0 <= x < width and 0 <= y < height:  # Check bounds
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            location = "at center" if distance == 0 else \
                      "on circle" if abs(distance - radius) < 0.1 else \
                      "inside circle" if distance < radius else \
                      "outside circle"
            print(f"  Point ({x}, {y}): distance={distance:.2f}, {location}")
        else:
            print(f"  Point ({x}, {y}): out of bounds")
    
    # Display the result
    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(test_img, cmap='gray')
        plt.title('Manual Circle Masking')
        
        # Add test points
        colors = ['red', 'yellow', 'blue', 'green']
        for (x, y), color in zip(test_points, colors):
            if 0 <= x < width and 0 <= y < height:  # Only plot if within bounds
                plt.plot(x, y, 'o', color=color, markersize=8)
        
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying manual example: {e}")


def create_synthetic_fiber_image():
    """
    Create a synthetic fiber optic image for testing
    """
    print("\nCreating synthetic fiber optic image for testing...")
    
    # Create a synthetic fiber optic-like image
    synthetic = np.ones((300, 300), dtype=np.uint8) * 200  # Background
    
    # Add cladding (medium gray) first
    cv2.circle(synthetic, (150, 150), 80, (150,), -1)
    
    # Add core (bright center) on top
    cv2.circle(synthetic, (150, 150), 40, (255,), -1)
    
    # Add some noise
    noise = np.random.normal(0, 5, synthetic.shape)
    synthetic = np.clip(synthetic + noise, 0, 255).astype(np.uint8)
    
    # Save synthetic image
    output_path = "synthetic_fiber.jpg"
    success = cv2.imwrite(output_path, synthetic)
    
    if success:
        print(f"Synthetic image saved as {output_path}")
        return output_path
    else:
        print("Failed to save synthetic image")
        return None


def main():
    """
    Main function with error handling and user interaction
    """
    print("Fiber Optic Masking Analysis")
    print("=" * 40)
    
    try:
        # Run the manual example first to show the mathematical concept
        manual_circle_masking_example()
        
        # Ask user for image choice
        print("\nChoose image source:")
        print("1. Create synthetic image")
        print("2. Use existing image file")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            image_path = create_synthetic_fiber_image()
            if image_path is None:
                print("Failed to create synthetic image")
                return
        else:
            image_path = input("Enter path to your fiber optic image: ").strip()
            # Remove quotes if present
            image_path = image_path.strip('"').strip("'")
        
        # Process the image
        results = separate_core_and_cladding(image_path)
        
        if results:
            print("\nProcessing complete!")
            print("Results dictionary contains:")
            print("  - core_image: Isolated core region")
            print("  - cladding_image: Isolated cladding region")
            print("  - core_mask: Binary mask for core")
            print("  - cladding_mask: Binary mask for cladding")
            print("  - core_circle: (x, y, radius) of core")
            print("  - cladding_circle: (x, y, radius) of cladding outer boundary")
            
            # Optional: Save results
            save_choice = input("\nSave masks to files? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                if results['core_mask'] is not None:
                    cv2.imwrite("core_mask.jpg", results['core_mask'])
                    cv2.imwrite("core_isolated.jpg", results['core_image'])
                    print("Core mask and image saved")
                
                if results['cladding_mask'] is not None:
                    cv2.imwrite("cladding_mask.jpg", results['cladding_mask'])
                    cv2.imwrite("cladding_isolated.jpg", results['cladding_image'])
                    print("Cladding mask and image saved")
        else:
            print("Processing failed!")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An error occurred: {e}")


# Main execution
if __name__ == "__main__":
    main()
