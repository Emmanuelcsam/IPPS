# -*- coding: utf-8 -*-
"""
Perfected Fiber Optic Core and Cladding Splitter

This script implements a robust, multi-stage pipeline to accurately detect
and segment the core and cladding regions of a fiber optic end-face image.
It overcomes the instability of simpler methods by using a robust voting
scheme for initial parameter estimation followed by a precise non-linear
least squares refinement.

Pipeline Stages:
1.  **Preprocessing & Edge Detection**: The image is pre-processed with a median
    blur to reduce noise, and Canny edge detection is used to extract
    high-confidence geometric features.
2.  **Robust Hypothesis Generation**:
    - Center Finding: A Hough-like voting scheme based on the perpendicular
      bisectors of random edge point pairs provides a highly stable center
      estimate, avoiding the pitfalls of 3-point RANSAC.
    - Radius Finding: A histogram of distances from the robust center to all
      edge points is created to find the two most likely radii.
3.  **High-Accuracy Refinement**: A non-linear least squares optimization
    (Trust Region Reflective with Cauchy loss) is performed to refine the
    center and radii, minimizing the true geometric distance to the edge
    points for maximum precision.
4.  **Masking & Output**: Final, precise masks for the core and cladding
    are generated based on the refined parameters. The script outputs the
    isolated core, isolated cladding, and a diagnostic plot showing the
    quality of the fit.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def find_robust_center(edge_points: np.ndarray, image_shape: tuple,
                         num_pairs: int = 5000, accumulator_bins: int = 500) -> tuple:
    """
    Finds a robust estimate of the circle's center using a Hough-like voting
    scheme based on perpendicular bisectors. This replaces unstable 3-point
    RANSAC methods. For any two points on a circle, the center must lie on
    their perpendicular bisector. By finding where many such bisectors
    intersect, we can robustly locate the center.

    Args:
        edge_points (np.ndarray): Array of (x, y) edge coordinates.
        image_shape (tuple): The (height, width) of the original image.
        num_pairs (int): The number of random point pairs to use for voting.
        accumulator_bins (int): The resolution of the voting accumulator.

    Returns:
        tuple: The estimated (x, y) coordinates of the center.
    """
    h, w = image_shape
    accumulator = np.zeros((accumulator_bins, accumulator_bins))

    n_points = len(edge_points)
    if n_points < 2:
        # Fallback to image center if not enough points
        return (w / 2, h / 2)

    # Randomly select pairs of points
    indices1 = np.random.choice(n_points, num_pairs, replace=True)
    indices2 = np.random.choice(n_points, num_pairs, replace=True)
    p1s = edge_points[indices1]
    p2s = edge_points[indices2]

    # Calculate midpoints and vectors for perpendicular bisectors
    midpoints = (p1s + p2s) / 2
    vectors = p2s - p1s

    # Avoid division by zero for horizontal lines
    vectors[np.abs(vectors[:, 1]) < 1e-6, 1] = 1e-6

    # Slopes of perpendicular bisectors (m_perp = -dx / dy)
    m_perp = -vectors[:, 0] / vectors[:, 1]
    # Intercepts (b = y - mx)
    b_perp = midpoints[:, 1] - m_perp * midpoints[:, 0]

    # Vote in the accumulator space
    x_range = np.linspace(0, w, accumulator_bins)
    for m, b in zip(m_perp, b_perp):
        # Calculate y values for each x along the bisector line
        y_vals = m * x_range + b

        # Convert to accumulator indices and filter valid ones
        x_indices = np.floor(x_range / w * accumulator_bins).astype(int)
        y_indices = np.floor(y_vals / h * accumulator_bins).astype(int)
        valid_mask = (y_indices >= 0) & (y_indices < accumulator_bins)
        
        # Increment the accumulator for valid points on the line
        np.add.at(accumulator, (y_indices[valid_mask], x_indices[valid_mask]), 1)

    # Find the bin with the most votes
    max_vote_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)

    # Convert accumulator index back to image coordinates for the center
    center_y = max_vote_idx[0] * h / accumulator_bins
    center_x = max_vote_idx[1] * w / accumulator_bins

    return center_x, center_y

def get_radii_from_histogram(edge_points: np.ndarray, center: tuple) -> tuple:
    """
    Calculates initial guesses for the two radii by creating a histogram of
    distances from the estimated center to all edge points. The two most
    frequent distances correspond to the core and cladding radii.

    Args:
        edge_points (np.ndarray): Array of (x, y) edge coordinates.
        center (tuple): The estimated (x, y) center.

    Returns:
        tuple: The estimated (radius1, radius2).
    """
    center_arr = np.array(center)
    distances = np.linalg.norm(edge_points - center_arr, axis=1)

    # Create a histogram of radii, finding the two most prominent peaks
    hist, bin_edges = np.histogram(distances, bins=200, range=(0, np.max(distances)))
    peak_indices = np.argsort(hist)[-2:] # Indices of two largest peaks

    # Calculate radii from the center of the peak bins
    r1_guess = bin_edges[peak_indices[0]] + (bin_edges[1] - bin_edges[0]) / 2
    r2_guess = bin_edges[peak_indices[1]] + (bin_edges[1] - bin_edges[0]) / 2

    return min(r1_guess, r2_guess), max(r1_guess, r2_guess)

def refine_fit(edge_points: np.ndarray, initial_params: list) -> np.ndarray:
    """
    Performs the final refinement using Non-Linear Least Squares to minimize
    the true geometric distance of edge points to the two-circle model.

    Args:
        edge_points (np.ndarray): Array of (x, y) edge coordinates.
        initial_params (list): Initial [cx, cy, r1, r2] guess.

    Returns:
        np.ndarray: The final, refined [cx, cy, r1, r2] parameters.
    """
    def residuals(params, points):
        """The objective function to minimize."""
        cx, cy, r1, r2 = params
        center = np.array([cx, cy])
        distances = np.linalg.norm(points - center, axis=1)
        # The error for each point is its distance to the *nearer* of the two circles
        err1 = np.abs(distances - r1)
        err2 = np.abs(distances - r2)
        return np.minimum(err1, err2)

    # Use Trust Region Reflective ('trf') algorithm for robust optimization.
    # The 'cauchy' loss function reduces the influence of outliers, making the
    # fit more stable against noisy edge points.
    result = least_squares(residuals, initial_params, args=(edge_points,), method='trf', loss='cauchy')

    # Ensure radii are ordered correctly in the final output
    final_params = result.x
    if final_params[2] > final_params[3]:
        final_params[2], final_params[3] = final_params[3], final_params[2]

    return final_params

def create_masks_and_split(image: np.ndarray, params: np.ndarray) -> tuple:
    """
    Creates final masks using the precise parameters and splits the core and cladding.
    Uses efficient NumPy broadcasting to avoid slow loops.

    Args:
        image (np.ndarray): The original grayscale image.
        params (np.ndarray): The final [cx, cy, r_core, r_cladding] parameters.

    Returns:
        tuple: (isolated_core, isolated_cladding, core_mask, cladding_mask)
    """
    h, w = image.shape
    cx, cy, r_core, r_cladding = params

    # Create coordinate grid efficiently using matrix operations
    y, x = np.mgrid[:h, :w]
    dist_sq = (x - cx)**2 + (y - cy)**2

    # Create binary masks based on the circle equations
    core_mask = (dist_sq <= r_core**2).astype(np.uint8)
    cladding_mask = ((dist_sq > r_core**2) & (dist_sq <= r_cladding**2)).astype(np.uint8)

    # Isolate the regions from the original image using the masks
    # The mask must be converted to 8-bit for bitwise_and
    isolated_core = cv2.bitwise_and(image, image, mask=core_mask)
    isolated_cladding = cv2.bitwise_and(image, image, mask=cladding_mask)

    # Crop images to their content for clean output
    coords_core = np.argwhere(core_mask > 0)
    if coords_core.size > 0:
        y_min, x_min = coords_core.min(axis=0)
        y_max, x_max = coords_core.max(axis=0)
        isolated_core = isolated_core[y_min:y_max+1, x_min:x_max+1]

    coords_cladding = np.argwhere(cladding_mask > 0)
    if coords_cladding.size > 0:
        y_min, x_min = coords_cladding.min(axis=0)
        y_max, x_max = coords_cladding.max(axis=0)
        isolated_cladding = isolated_cladding[y_min:y_max+1, x_min:x_max+1]

    return isolated_core, isolated_cladding

def generate_diagnostic_plot(original_image, edge_points, params, output_path):
    """Generates and saves a diagnostic plot showing the fitted circles."""
    cx, cy, r_core, r_cladding = params

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    # Plot the refined circles
    circle1 = plt.Circle((cx, cy), r_core, color='lime', fill=False, linewidth=2, label='Fitted Core')
    circle2 = plt.Circle((cx, cy), r_cladding, color='cyan', fill=False, linewidth=2, label='Fitted Cladding')
    plt.gca().add_artist(circle1)
    plt.gca().add_artist(circle2)

    # Plot the center
    plt.plot(cx, cy, 'r+', markersize=15, markeredgewidth=3, label='Detected Center')

    # Plot the edge points used for fitting
    plt.scatter(edge_points[:, 0], edge_points[:, 1], s=1, c='red', alpha=0.3, label='Canny Edge Points')

    plt.title(f'Perfected Geometric Fit for {os.path.basename(output_path)}')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def perfected_fiber_splitter(image_path: str, output_dir: str = 'perfected_output'):
    """
    Main processing pipeline to detect, refine, and split fiber optic core and cladding.

    Args:
        image_path (str): Path to the fiber optic end-face image.
        output_dir (str): Directory to save the results.
    """
    print(f"\n--- Perfected Pipeline commencing for: {image_path} ---")
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # --- STAGE 1: Pre-processing and Edge Detection ---
    blurred_image = cv2.medianBlur(gray_image, 5)
    edges = cv2.Canny(blurred_image, 50, 150)
    edge_points = np.argwhere(edges).astype(float)[:, ::-1] # Get (x, y) points

    if len(edge_points) < 50:
        print("Error: Not enough edge points detected. Check Canny parameters or image quality.")
        return

    print(f"Stage 1: Extracted {len(edge_points)} edge points.")

    # --- STAGE 2: Robust Hypothesis Generation ---
    center_guess = find_robust_center(edge_points, gray_image.shape)
    r1_guess, r2_guess = get_radii_from_histogram(edge_points, center_guess)

    initial_guess = [center_guess[0], center_guess[1], r1_guess, r2_guess]
    print(f"Stage 2: Robust initial guess -> Center:({initial_guess[0]:.2f}, {initial_guess[1]:.2f}), Radii:({initial_guess[2]:.2f}, {initial_guess[3]:.2f})")

    # --- STAGE 3: Final Refinement ---
    final_params = refine_fit(edge_points, initial_guess)
    cx, cy, r_core, r_cladding = final_params
    print(f"Stage 3: Final refined parameters -> Center:({cx:.4f}, {cy:.4f}), Radii:({r_core:.4f}, {r_cladding:.4f})")

    # --- STAGE 4: Masking, Splitting, and Saving ---
    core_img, cladding_img = create_masks_and_split(gray_image, final_params)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Save the diagnostic visualization
    plot_path = os.path.join(output_dir, f"{base_filename}_diagnostic_fit.png")
    generate_diagnostic_plot(original_image, edge_points, final_params, plot_path)

    # Save the final split images
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_core_split.png"), core_img)
    cv2.imwrite(os.path.join(output_dir, f"{base_filename}_cladding_split.png"), cladding_img)
    print(f"Stage 4: Successfully saved results to '{output_dir}'")

    return {
        'center': (cx, cy),
        'core_radius': r_core,
        'cladding_radius': r_cladding
    }

if __name__ == '__main__':
    # --- IMPORTANT ---
    # You must provide a valid path to your fiber optic image here.
    # Example: test_image_path = 'path/to/your/image.jpg'
    test_image_path = r'C:\Users\Saem1001\Documents\GitHub\IPPS\App\images\img63.jpg' #<-- CHANGE THIS

    if os.path.exists(test_image_path):
        perfected_fiber_splitter(test_image_path)
    else:
        print(f"Test image not found. Please update the 'test_image_path' variable in the script.")
        # Create a dummy image for demonstration if no image is found
        print("Creating a dummy image for demonstration purposes.")
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Draw concentric circles to simulate a fiber
        cv2.circle(dummy_img, (250, 260), 150, (180, 180, 180), -1) # Cladding
        cv2.circle(dummy_img, (250, 260), 60, (230, 230, 230), -1)  # Core
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 10, dummy_img.shape).astype(np.uint8)
        dummy_img = cv2.add(dummy_img, noise)

        dummy_path = 'dummy_fiber_image.png'
        cv2.imwrite(dummy_path, dummy_img)
        perfected_fiber_splitter(dummy_path)