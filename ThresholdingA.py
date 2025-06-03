#Thresholding
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Fiber Optic End Face Segmentation using Thresholding
# Goal: Identify core, cladding, and ferrule regions
base_path = 'C:/Users/Saem1001/Documents/GitHub/OpenCV-Practice/'
image_path = base_path + '19700103045135-J67690-FT41.jpg'
 
def segment_fiber_optic(image_path):
    """
    Segment fiber optic end face into core, cladding, and ferrule regions
    using various thresholding techniques.
    """
    # Read the image
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    # Apply Gaussian blur to reduce noise
    img_blur = cv.GaussianBlur(img, (5, 5), 0)
    
    # Method 1: Simple Thresholding
    # Threshold for core (brightest region)
    ret1, core_simple = cv.threshold(img_blur, 200, 255, cv.THRESH_BINARY)
    
    # Threshold for cladding (middle brightness)
    ret2, temp = cv.threshold(img_blur, 100, 255, cv.THRESH_BINARY)
    ret3, core_inv = cv.threshold(img_blur, 200, 255, cv.THRESH_BINARY_INV)
    cladding_simple = cv.bitwise_and(temp, core_inv)
    
    # Threshold for ferrule (darkest region)
    ret4, ferrule_simple = cv.threshold(img_blur, 100, 255, cv.THRESH_BINARY_INV)
    
    # Method 2: Otsu's Thresholding
    # First threshold to separate ferrule from fiber
    ret5, ferrule_otsu = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Mask to get only fiber region (core + cladding)
    fiber_region = cv.bitwise_not(ferrule_otsu)
    fiber_masked = cv.bitwise_and(img_blur, fiber_region)
    
    # Second threshold on fiber region to separate core from cladding
    # Only process non-zero pixels
    fiber_pixels = fiber_masked[fiber_masked > 0]
    if len(fiber_pixels) > 0:
        threshold_value = cv.threshold(fiber_pixels, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[0]
        ret6, core_otsu = cv.threshold(fiber_masked, threshold_value, 255, cv.THRESH_BINARY)
    else:
        core_otsu = np.zeros_like(img)
    
    # Calculate cladding region
    cladding_otsu = cv.bitwise_and(fiber_region, cv.bitwise_not(core_otsu))
    
    # Method 3: Adaptive Thresholding
    # This method works well for non-uniform illumination
    adaptive_thresh = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv.THRESH_BINARY, 11, 2)
    
    # Create colored visualization
    # Simple thresholding result
    result_simple = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    result_simple[core_simple > 0] = [255, 255, 0]  # Yellow for core
    result_simple[cladding_simple > 0] = [0, 255, 255]  # Cyan for cladding
    result_simple[ferrule_simple > 0] = [128, 128, 128]  # Gray for ferrule
    
    # Otsu's thresholding result
    result_otsu = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    result_otsu[core_otsu > 0] = [255, 255, 0]  # Yellow for core
    result_otsu[cladding_otsu > 0] = [0, 255, 255]  # Cyan for cladding
    result_otsu[ferrule_otsu > 0] = [128, 128, 128]  # Gray for ferrule
    
    # Plot results
    titles = ['Original Image', 'Simple Thresholding', "Otsu's Thresholding",
              'Core (Simple)', 'Cladding (Simple)', 'Ferrule (Simple)',
              'Core (Otsu)', 'Cladding (Otsu)', 'Ferrule (Otsu)']
    
    images = [img, result_simple, result_otsu,
              core_simple, cladding_simple, ferrule_simple,
              core_otsu, cladding_otsu, ferrule_otsu]
    
    plt.figure(figsize=(15, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        if i < 3 and i > 0:  # Colored results
            plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
        else:  # Grayscale images
            plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return result_simple, result_otsu

def analyze_fiber_dimensions(image_path):
    """
    Analyze and measure the dimensions of fiber optic components
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    # Apply Gaussian blur
    img_blur = cv.GaussianBlur(img, (5, 5), 0)
    
    # Use Otsu's method to find optimal thresholds
    ret1, ferrule = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    fiber_region = cv.bitwise_not(ferrule)
    fiber_masked = cv.bitwise_and(img_blur, fiber_region)
    
    # Find core
    fiber_pixels = fiber_masked[fiber_masked > 0]
    if len(fiber_pixels) > 0:
        threshold_value = cv.threshold(fiber_pixels, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[0]
        ret2, core = cv.threshold(fiber_masked, threshold_value, 255, cv.THRESH_BINARY)
    else:
        core = np.zeros_like(img)
    
    # Find contours
    contours_core, _ = cv.findContours(core, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_fiber, _ = cv.findContours(fiber_region, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create visualization
    result = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    # Measurements
    measurements = {}
    
    # Core measurements
    if contours_core:
        largest_core = max(contours_core, key=cv.contourArea)
        cv.drawContours(result, [largest_core], -1, (255, 255, 0), 2)
        
        # Fit circle to core
        (x, y), radius = cv.minEnclosingCircle(largest_core)
        center = (int(x), int(y))
        radius = int(radius)
        cv.circle(result, center, radius, (0, 255, 0), 1)
        
        measurements['core_diameter'] = radius * 2
        measurements['core_center'] = center
    
    # Cladding measurements
    if contours_fiber:
        largest_fiber = max(contours_fiber, key=cv.contourArea)
        cv.drawContours(result, [largest_fiber], -1, (0, 255, 255), 2)
        
        # Fit circle to fiber (core + cladding)
        (x, y), radius = cv.minEnclosingCircle(largest_fiber)
        center = (int(x), int(y))
        radius = int(radius)
        cv.circle(result, center, radius, (0, 255, 0), 1)
        
        measurements['fiber_diameter'] = radius * 2
        measurements['cladding_diameter'] = radius * 2
    
    # Display measurements
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title('Fiber Optic Analysis')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.8, 'Measurements (pixels):', fontsize=12, fontweight='bold')
    y_pos = 0.6
    for key, value in measurements.items():
        if isinstance(value, tuple):
            plt.text(0.1, y_pos, f'{key}: {value}', fontsize=10)
        else:
            plt.text(0.1, y_pos, f'{key}: {value:.1f}', fontsize=10)
        y_pos -= 0.15
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return measurements

# Example usage
if __name__ == "__main__":
    # Note: Replace 'fiber_optic.png' with your actual fiber optic image
    # The image should be a grayscale image of a fiber optic end face
    
    # Example of creating a synthetic fiber optic image for demonstration
    # In practice, use your actual fiber optic microscope image
    def create_synthetic_fiber_image():
        """Create a synthetic fiber optic end face image for demonstration"""
        img = np.zeros((400, 400), dtype=np.uint8)
        
        # Draw ferrule (background)
        img[:, :] = 50
        
        # Draw cladding
        cv.circle(img, (200, 200), 80, 150, -1)
        
        # Draw core
        cv.circle(img, (200, 200), 25, 250, -1)
        
        # Add some noise
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        return img
    
    # Create synthetic image for demonstration
    synthetic_img = create_synthetic_fiber_image()
    cv.imwrite('synthetic_fiber.png', synthetic_img)
    
    print("Fiber Optic End Face Segmentation")
    print("=================================")
    print()
    
    # Segment the fiber optic components
    print("1. Performing segmentation...")
    result_simple, result_otsu = segment_fiber_optic('synthetic_fiber.png')
    
    # Analyze dimensions
    print("\n2. Analyzing fiber dimensions...")
    measurements = analyze_fiber_dimensions('synthetic_fiber.png')
    
    print("\nSegmentation complete!")

