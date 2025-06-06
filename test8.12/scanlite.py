import cv2
import numpy as np
import matplotlib.pyplot as plt

# #####################################################################
# STAGE 1: IMAGE ACQUISITION AND PRE-PROCESSING
# #####################################################################
def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, and applies Gaussian blur for denoising.
    This corresponds to Stage 1 of the inspection pipeline. 

    Args:
        image_path (str): The file path to the fiber optic end face image.

    Returns:
        tuple: A tuple containing the original color image and the preprocessed grayscale image.
    """
    # Load Image: Read the image file.
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert to Grayscale: Operations work best on single-channel images.
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Denoise: Apply a Gaussian blur to reduce noise from image acquisition. 
    # A 5x5 kernel is a common choice for moderate smoothing.
    preprocessed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    print("Stage 1: Image Pre-processing Complete.")
    return original_image, preprocessed_image

# #####################################################################
# STAGE 2: LOCATE FIBER AND DEFINE REGIONS
# #####################################################################
def locate_fiber_and_define_zones(image, gray_image):
    """
    Automatically finds the fiber's outer boundary using Hough Circle Transform
    and defines the core and cladding zones based on standard dimensions.
    This corresponds to Stage 2 of the inspection pipeline. 

    Args:
        image (np.array): The original color image for displaying results.
        gray_image (np.array): The preprocessed grayscale image for detection.

    Returns:
        tuple: A tuple containing:
               - The display image with zones drawn on it.
               - A dictionary with the center and radii (in pixels) of the zones.
               - A mask for the combined core and cladding area.
    """
    display_image = image.copy()
    h, w = image.shape[:2]
    zones = {}
    
    # Automatic Circle Detection: Use HoughCircles to find the cladding. 
    # These parameters (minDist, param1, param2) may need tuning for different image sets.
    # minDist: Minimum distance between detected centers.
    # param1: Higher threshold for the Canny edge detector.
    # param2: Accumulator threshold for circle detection.
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=w//2,
                               param1=120, param2=50, minRadius=w//4, maxRadius=w//2)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Use the most prominent circle found
        c = circles[0, 0]
        center = (c[0], c[1])
        cladding_radius = c[2]

        # Define other zones based on typical fiber geometry.
        # These ratios are illustrative. Real ratios depend on fiber type (e.g., LC).
        # Per IEC 61300-3-35, zones are defined by diameter.
        # A: Core (0-25µm), B: Cladding (25-120µm), C: Adhesive (120-130µm), D: Contact (130-250µm)
        # We will approximate this based on the detected cladding.
        core_radius = int(cladding_radius * (25 / 125)) # Approx. core diameter relative to cladding
        adhesive_radius = int(cladding_radius * (130 / 125))
        
        zones = {
            'center': center,
            'core_radius': core_radius,
            'cladding_radius': cladding_radius,
            'adhesive_radius': adhesive_radius
        }

        # Display Core and Cladding Regions
        # Core (Yellow)
        cv2.circle(display_image, center, core_radius, (0, 255, 255), 2)
        # Cladding (Blue) 
        cv2.circle(display_image, center, cladding_radius, (255, 0, 0), 2)
        # Adhesive Zone (Green)
        cv2.circle(display_image, center, adhesive_radius, (0, 255, 0), 2)

        # Display Diameters
        cladding_diameter_px = cladding_radius * 2
        cv2.putText(display_image, f"Cladding Dia: {cladding_diameter_px} px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print("Stage 2: Fiber Location and Zone Definition Complete.")
        
        # Create a mask for the area of interest (core + cladding)
        # Scratch inspection is only required in these zones 
        analysis_mask = np.zeros(gray_image.shape, dtype="uint8")
        cv2.circle(analysis_mask, center, cladding_radius, 255, -1)
        
        return display_image, zones, analysis_mask

    else:
        print("Stage 2: Failed. No fiber circle detected.")
        # Return empty results if no circle is found
        return display_image, None, None

# #####################################################################
# STAGE 3A: REGION-BASED DEFECTS (DO2MR METHOD) - CORRECTED
# #####################################################################
def run_do2mr_method(gray_image, analysis_mask):
    """
    Implements the DO2MR ("Difference of Min-Max Ranking") filter to find
    region-based defects like dirt, pits, and oil.

    Args:
        gray_image (np.array): The preprocessed grayscale image.
        analysis_mask (np.array): Mask to focus the analysis on the fiber face.

    Returns:
        np.array: A binary mask showing detected region-based defects.
    """
    print("Stage 3A: Running DO2MR for Region-Based Defects...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    img_max = cv2.dilate(gray_image, kernel)
    img_min = cv2.erode(gray_image, kernel)
    residual_map = cv2.subtract(img_max, img_min)

    # Calculate mu and sigma on the pixels within the fiber face.
    mu, sigma = cv2.meanStdDev(residual_map, mask=analysis_mask)
    
    # --- FIX IS HERE ---
    # Extract the numerical values from the arrays before calculation.
    gamma = 3.0 # Hyper-parameter, tune as needed.
    threshold_value = mu[0][0] + gamma * sigma[0][0]
    
    # Use the single numerical threshold_value in the function.
    _, defect_mask = cv2.threshold(residual_map, threshold_value, 255, cv2.THRESH_BINARY)
    # --- END FIX ---

    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, opening_kernel)
    
    defect_mask = cv2.bitwise_and(defect_mask, analysis_mask)

    print("Stage 3A: DO2MR Complete.")
    return defect_mask

# #####################################################################
# STAGE 3B: SCRATCH DEFECTS (LEI METHOD)
# #####################################################################
def run_lei_method(gray_image, analysis_mask):
    """
    Implements the LEI ("Linear Enhancement Inspector") method to find
    low-contrast linear scratches at various angles. 

    Args:
        gray_image (np.array): The preprocessed grayscale image.
        analysis_mask (np.array): Mask to focus the analysis on the fiber face.

    Returns:
        np.array: A binary mask showing detected scratch defects.
    """
    print("Stage 3B: Running LEI for Scratch Defects...")
    # Image Enhancement: Use histogram equalization to improve contrast for faint scratches. 
    enhanced_image = cv2.equalizeHist(gray_image)

    # Scratch Searching and Segmentation
    # The paper uses a special linear detector. We will implement this by creating
    # kernels that represent lines at different orientations.
    final_scratch_map = np.zeros(enhanced_image.shape, dtype=np.uint8)
    
    # Search in 15-degree increments from 0 to 180 
    for angle in range(0, 180, 15):
        # Create a line kernel. Length 15, thickness 1.
        # This kernel enhances linear features.
        kernel_len = 15
        line_kernel = np.zeros((kernel_len, kernel_len), dtype=np.float32)
        
        # Define the line in the center of the kernel
        # The paper's detector has negative weights on the side, we simplify by
        # enhancing the line and thresholding. This is a common implementation.
        line_kernel[int((kernel_len-1)/2), :] = 1
        
        # Rotate the kernel to the desired angle
        center = (int((kernel_len-1)/2), int((kernel_len-1)/2))
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_kernel = cv2.warpAffine(line_kernel, rot_mat, (kernel_len, kernel_len))
        
        # Normalize the kernel
        rotated_kernel /= cv2.sumElems(rotated_kernel)[0]
        
        # Apply the filter to get a response map for this orientation
        response_map = cv2.filter2D(enhanced_image, -1, rotated_kernel)

        # Scratch Segmentation: Threshold each response map individually.
        # This threshold will be sensitive and may need tuning.
        _, thresholded_map = cv2.threshold(response_map, 140, 255, cv2.THRESH_BINARY)
        
        # Result Synthesization: Combine maps with a bitwise OR operation. 
        final_scratch_map = cv2.bitwise_or(final_scratch_map, thresholded_map)

    # Apply analysis mask to restrict search to the fiber face
    final_scratch_map = cv2.bitwise_and(final_scratch_map, analysis_mask)

    print("Stage 3B: LEI Complete.")
    return final_scratch_map


# #####################################################################
# STAGE 4: ANALYSIS, TOOLING, AND LEARNING
# #####################################################################
def analyze_and_display_results(display_image, region_defects, scratch_defects, zones):
    """
    Analyzes the detected defects within each zone, counts them, and overlays
    the results on the display image for visualization.

    Args:
        display_image (np.array): The image with zones drawn on it.
        region_defects (np.array): Binary mask of region-based defects.
        scratch_defects (np.array): Binary mask of scratch defects.
        zones (dict): Dictionary containing the geometry of the fiber zones.

    Returns:
        np.array: The final image with all defects highlighted.
    """
    print("Stage 4: Analyzing and Visualizing Results...")
    # Use different colors to highlight different defect types
    # Blue for Region-based defects (DO2MR)
    display_image[region_defects == 255] = [255, 100, 100]
    # Red for Scratches (LEI)
    display_image[scratch_defects == 255] = [100, 100, 255]

    # Analyze Each Region Separately
    if zones:
        # Create masks for each specific zone (e.g., core)
        core_mask = np.zeros(region_defects.shape, dtype="uint8")
        cv2.circle(core_mask, zones['center'], zones['core_radius'], 255, -1)
        
        # Isolate defects within the core using a bitwise AND
        core_region_defects = cv2.bitwise_and(region_defects, core_mask)
        core_scratch_defects = cv2.bitwise_and(scratch_defects, core_mask)

        # Use findContours to count the number of defects in the core
        # For region defects
        contours_region, _ = cv2.findContours(core_region_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # For scratches
        contours_scratch, _ = cv2.findContours(core_scratch_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        num_core_defects = len(contours_region) + len(contours_scratch)
        
        # Display the count on the image
        text = f"Defects in Core: {num_core_defects}"
        cv2.putText(display_image, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print(f"Analysis complete. Found {num_core_defects} defects in the core.")
        
    return display_image

# --- Main Execution ---
if __name__ == '__main__':
    # You can download a sample image from the paper's dataset link or use your own.
    # Figure 4(a) 'dirt' is a good example for DO2MR.
    # Figure 4(c) 'pit and chip' is also good for DO2MR.
    # Figure 4(d) 'scratch' is the target for LEI.
    # The program will run both detectors on the input image.
    
    # Replace with the path to your fiber image.
    # You can find the dataset mentioned in the paper here:
    # https://pan.baidu.com/s/13b3JqBcUlaookvcyOkqxsQ 
    # For this example, let's assume we have an image named 'fiber_scratch.png'.
    # I will create a dummy image if one is not found.
    IMAGE_PATH = 'fiber_image.png'

    try:
        # Stage 1: Load and Pre-process
        original_image, gray_image = preprocess_image(IMAGE_PATH)
    except FileNotFoundError:
        print(f"'{IMAGE_PATH}' not found. Creating a dummy test image.")
        # Create a dummy image for demonstration if no image is available
        sz = 600
        original_image = np.full((sz, sz, 3), (200, 200, 200), dtype=np.uint8)
        # Draw a "fiber"
        cv2.circle(original_image, (sz//2, sz//2), 200, (150, 150, 150), -1) # Ferrule
        cv2.circle(original_image, (sz//2, sz//2), 180, (50, 50, 50), -1)   # Cladding
        # Add a "scratch" defect
        cv2.line(original_image, (250, 250), (400, 400), (80, 80, 80), 3)
        # Add a "dirt" defect
        cv2.circle(original_image, (450, 200), 10, (20, 20, 20), -1)
        cv2.imwrite(IMAGE_PATH, original_image)
        original_image, gray_image = preprocess_image(IMAGE_PATH)
        

    # Stage 2: Locate Fiber and Define Zones
    display_image, zones, analysis_mask = locate_fiber_and_define_zones(original_image, gray_image)

    if zones: # Proceed only if a fiber was successfully located
        # Stage 3: Defect Detection
        # A. Run DO2MR for region-based defects
        region_defects_mask = run_do2mr_method(gray_image, analysis_mask)
        # B. Run LEI for scratch defects
        scratch_defects_mask = run_lei_method(gray_image, analysis_mask)

        # Stage 4: Analyze and Display
        final_result_image = analyze_and_display_results(display_image, region_defects_mask, scratch_defects_mask, zones)
    
        # --- Display all results using Matplotlib ---
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.title('1. Original Image')
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title('2. Zones Identified')
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title('3a. DO2MR Result (Region Defects)')
        plt.imshow(region_defects_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.title('3b. LEI Result (Scratch Defects)')
        plt.imshow(scratch_defects_mask, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('4. Final Analysis')
        plt.imshow(cv2.cvtColor(final_result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()