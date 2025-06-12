import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def find_iteratively_refined_center(image, search_radius=3):
    """
    Finds the fiber center with maximum accuracy by performing an iterative
    local search to find the center point that maximizes the sharpness of the
    core/cladding intensity drop. This function is retained for its robustness.
    """
    # [This function's code is identical to the previous version and is retained for brevity]
    # [It robustly finds the best center, which is crucial for the next steps]
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M = cv2.moments(thresh_image)
    if M["m00"] != 0:
        initial_cx, initial_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        h, w = image.shape
        initial_cx, initial_cy = w // 2, h // 2

    best_center = (initial_cx, initial_cy)
    max_sharpness = 0  # We want the most negative derivative

    h, w = image.shape
    y_coords, x_coords = np.indices((h, w))

    for cy_candidate in range(initial_cy - search_radius, initial_cy + search_radius + 1):
        for cx_candidate in range(initial_cx - search_radius, initial_cx + search_radius + 1):
            r = np.sqrt((x_coords - cx_candidate)**2 + (y_coords - cy_candidate)**2).astype(int)
            # Efficiently calculate profile for sharpness check
            tbin = np.bincount(r.ravel(), image.ravel())
            nr = np.bincount(r.ravel())
            radial_profile = np.divide(tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr != 0)
            
            profile_segment = radial_profile[:w // 3]
            if len(profile_segment) < 20: continue

            derivative = np.gradient(cv2.GaussianBlur(profile_segment, (5, 1), 0).flatten())
            sharpness = np.min(derivative) # Most negative value

            if sharpness < max_sharpness:
                max_sharpness = sharpness
                best_center = (cx_candidate, cy_candidate)
                
    return best_center

def fiber_model_function(r, r_core, w_core, r_cladding, w_cladding, I_core, I_cladding, I_ferrule):
    """
    A mathematical model of the fiber's radial intensity profile using tanh functions
    to create smooth, sigmoidal steps. This is our idealized equation.
    
    Args:
        r (np.array): Array of radii.
        r_core (float): The center radius of the core-cladding transition.
        w_core (float): The width (steepness) of the core-cladding transition.
        r_cladding (float): The center radius of the cladding-ferrule transition.
        w_cladding (float): The width (steepness) of the cladding-ferrule transition.
        I_core (float): The intensity level of the core.
        I_cladding (float): The intensity level of the cladding.
        I_ferrule (float): The intensity level of the ferrule.

    Returns:
        np.array: The modeled intensity profile.
    """
    # Term 1: Models the drop from core to cladding
    core_to_cladding_term = (I_core - I_cladding) / 2 * (1 - np.tanh((r - r_core) / w_core))
    
    # Term 2: Models the rise from cladding to ferrule
    cladding_to_ferrule_term = (I_ferrule - I_cladding) / 2 * (1 + np.tanh((r - r_cladding) / w_cladding))

    return core_to_cladding_term + cladding_to_ferrule_term + I_cladding


def analyze_profile_with_model_fit(image, center):
    """
    Analyzes the radial profile by fitting a sophisticated mathematical model to it,
    extracting highly accurate parameters for the core and cladding radii.
    """
    # --- 1. Calculate the real-world radial data from the pixel matrix ---
    cx, cy = center
    h, w = image.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Bin pixels by integer radius and calculate average intensity
    r_int = r.astype(int)
    tbin = np.bincount(r_int.ravel(), image.ravel())
    nr = np.bincount(r_int.ravel())
    radial_profile = np.divide(tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr != 0)
    max_radius = min(cx, cy, w - cx, h - cy)
    radii_x_axis = np.arange(max_radius)
    profile_y_axis = radial_profile[:max_radius]

    # --- 2. Make intelligent initial guesses for the model parameters (crucial for convergence) ---
    # We use the derivative method as a starting point for the optimization.
    derivative = np.gradient(cv2.GaussianBlur(profile_y_axis, (5, 1), 0).flatten())
    
    guess_r_core = np.argmin(derivative[5:]) + 5
    guess_r_cladding = np.argmax(derivative[guess_r_core:]) + guess_r_core
    guess_I_core = np.mean(profile_y_axis[:guess_r_core])
    guess_I_cladding = np.min(profile_y_axis[guess_r_core:guess_r_cladding])
    guess_I_ferrule = np.mean(profile_y_axis[guess_r_cladding + 5:])

    # Initial parameter guesses [r_core, w_core, r_cladding, w_cladding, I_core, I_cladding, I_ferrule]
    p0 = [guess_r_core, 5.0, guess_r_cladding, 5.0, guess_I_core, guess_I_cladding, guess_I_ferrule]

    # --- 3. Perform the Non-Linear Least Squares Fit (The "Magic") ---
    try:
        # We provide the model, the data (x, y), the initial guesses (p0), and bounds.
        bounds = ([0, 0, 0, 0, 0, 0, 0], [max_radius, max_radius, max_radius, max_radius, 255, 255, 255])
        popt, _ = curve_fit(fiber_model_function, radii_x_axis, profile_y_axis, p0=p0, bounds=bounds, maxfev=5000)
        
        # The optimized parameters are our highly accurate results
        r_core_fit, _, r_cladding_fit, _, _, _, _ = popt
        
        # Generate the fitted curve for plotting
        fitted_curve = fiber_model_function(radii_x_axis, *popt)

    except RuntimeError:
        print("Warning: Curve fit failed. Falling back to derivative method.")
        r_core_fit, r_cladding_fit = guess_r_core, guess_r_cladding
        fitted_curve = None

    return int(round(r_core_fit)), int(round(r_cladding_fit)), profile_y_axis, fitted_curve

# The main pipeline function now uses these new analysis tools.
# The `create_and_apply_masks` and `crop_to_content` functions remain the same.
def create_and_apply_masks(image, center, r_core, r_cladding):
    # This function is correct and remains unchanged.
    cx, cy = center; h, w = image.shape; y, x = np.indices((h, w))
    dist_sq = (x - cx)**2 + (y - cy)**2
    core_mask = (dist_sq <= r_core**2).astype(np.uint8) * 255
    cladding_mask = ((dist_sq > r_core**2) & (dist_sq <= r_cladding**2)).astype(np.uint8) * 255
    isolated_core = cv2.bitwise_and(image, image, mask=core_mask)
    isolated_cladding = cv2.bitwise_and(image, image, mask=cladding_mask)
    return isolated_core, isolated_cladding, core_mask, cladding_mask

def crop_to_content(image, mask):
    # This function is correct and remains unchanged.
    coords = np.argwhere(mask > 0)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
        return image[y_min:y_max + 1, x_min:x_max + 1]
    return image

def process_fiber_image_model_fit(image_path, output_dir='output_model_fit'):
    print(f"\n--- Processing with Mathematical Model Fit: {image_path} ---")
    if not os.path.exists(image_path): print(f"Error: Not found: {image_path}"); return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    center = find_iteratively_refined_center(gray_image, search_radius=3)
    print(f"Refined Center: {center}")

    r_core, r_cladding, profile, fitted_curve = analyze_profile_with_model_fit(gray_image, center)
    print(f"Model Fit Radii -> Core: {r_core}px, Cladding: {r_cladding}px")
    
    core_img, cladding_img, core_mask, cladding_mask = create_and_apply_masks(gray_image, center, r_core, r_cladding)
    
    cropped_core = crop_to_content(core_img, core_mask)
    cropped_cladding = crop_to_content(cladding_img, cladding_mask)

    # --- Save Diagnostic Plot with the Fitted Model Curve ---
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    plt.figure(figsize=(12, 6))
    plt.plot(profile, 'b.', markersize=4, label='Actual Pixel Data')
    if fitted_curve is not None:
        plt.plot(fitted_curve, 'r-', linewidth=2, label='Fitted Mathematical Model')
    plt.axvline(x=r_core, color='g', linestyle='--', label=f'Core Radius ({r_core}px)')
    plt.axvline(x=r_cladding, color='m', linestyle='--', label=f'Cladding Radius ({r_cladding}px)')
    plt.xlabel('Radius (pixels from center)'); plt.ylabel('Pixel Intensity'); plt.grid(True)
    plt.title(f'Mathematical Model Fit for {os.path.basename(image_path)}'); plt.legend()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_model_fit_plot.png"))
    plt.close()

    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_core.png"), cropped_core)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_cladding.png"), cropped_cladding)
    print(f"Successfully saved model fit results to '{output_dir}'")


if __name__ == '__main__':
    image_filenames = [
        r'C:\Users\Saem1001\Documents\GitHub\IPPS\App\images\img63.jpg'
    ]
    for filename in image_filenames:
        process_fiber_image_model_fit(filename)
