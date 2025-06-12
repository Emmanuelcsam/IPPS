#!/usr/bin/env python3
#!/usr/bin/env python3
"""Adaptive Thresholding"""
import cv2
import numpy as np

# STEP 1: All your original code is now inside this one function.
def run_script_logic(image):
    # Your original code used a grayscale image, so let's ensure the input is grayscale.
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply different thresholding methods

    # 1. Simple global threshold
    _, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 2. Otsu's threshold (automatic threshold selection)
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Adaptive threshold - Gaussian weighted mean
    adaptive_gaussian = cv2.adaptiveThreshold(image, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, -2)

    # 4. Adaptive threshold - Mean
    adaptive_mean = cv2.adaptiveThreshold(image, 255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 15, -2)

    # Display results
    cv2.imshow('Original', image)
    cv2.imshow('Global Threshold', global_thresh)
    cv2.imshow('Otsu Threshold', otsu_thresh)
    cv2.imshow('Adaptive Gaussian', adaptive_gaussian)
    cv2.imshow('Adaptive Mean', adaptive_mean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save results
    cv2.imwrite('global_threshold.png', global_thresh)
    cv2.imwrite('otsu_threshold.png', otsu_thresh)
    cv2.imwrite('adaptive_gaussian.png', adaptive_gaussian)
    cv2.imwrite('adaptive_mean.png', adaptive_mean)

    return adaptive_gaussian

# =========================================================================
# UNIVERSAL UI ADAPTER: PASTE THIS AT THE END OF EVERY SCRIPT
# =========================================================================
def process_image(image: np.ndarray) -> np.ndarray:
    """
    This is the standard function the UI will call.
    It runs the main logic of this script.
    """
    # This line calls the function you created in Step 1.
    # It passes the UI's image to your logic and returns the result.
    return run_script_logic(image)

if __name__ == '__main__':
    # This part lets you run the script by itself for testing.
    # It loads your original hardcoded image and passes it to your logic function.
    
    # You can keep your original image path here for easy testing.
    image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
    original_image = cv2.imread(image_path)
    
    if original_image is not None:
        # When running standalone, we call the logic directly.
        # You could add your original cv2.imshow() calls here to see all results.
        final_result = run_script_logic(original_image)
        
        cv2.imshow('Original Test Image', original_image)
        cv2.imshow('Final Result (as seen in UI)', final_result)
        
        print("Press any key to close the windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Could not load the test image from {image_path}")
# =========================================================================