"""
Unified High-Precision Fiber Optic Core and Cladding Splitter

This script combines the best strategies from all provided implementations:
- Robust preprocessing with wavelet denoising and median filtering
- Phase congruency edge detection for illumination invariance
- Perpendicular bisector voting for stable center detection
- Multi-method radius detection with histogram and Fourier analysis
- Robust non-linear refinement with geometric constraints
"""

import cv2
import numpy as np
import os
import pywt
from scipy.optimize import least_squares
from scipy import signal, ndimage
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

def wavelet_denoise(image):
    """
    Advanced wavelet denoising using stationary wavelet transform.
    Adapted from fiber_optic_phd.py for superior noise reduction.
    """
    # Normalize image
    img_float = image.astype(np.float64) / 255.0
    
    # Stationary wavelet transform
    coeffs = pywt.swt2(img_float, 'db4', level=3)
    
    # Adaptive thresholding
    denoised_coeffs = []
    for level, (cA, (cH, cV, cD)) in enumerate(coeffs):
        # Estimate noise using median absolute deviation
        sigma = np.median(np.abs(cD)) / 0.6745
        
        # Adaptive threshold
        threshold = sigma * np.sqrt(2 * np.log(cD.size))
        
        # Soft thresholding
        cH = np.sign(cH) * np.maximum(np.abs(cH) - threshold, 0)
        cV = np.sign(cV) * np.maximum(np.abs(cV) - threshold, 0)
        cD = np.sign(cD) * np.maximum(np.abs(cD) - threshold, 0)
        
        denoised_coeffs.append((cA, (cH, cV, cD)))
    
    # Reconstruct
    denoised = pywt.iswt2(denoised_coeffs, 'db4')
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)

def phase_congruency_edges(image):
    """
    Phase congruency edge detection for illumination invariance.
    Simplified from fiber_optic_phd.py for efficiency.
    """
    # Apply Gabor filters at multiple scales and orientations
    edges = np.zeros_like(image, dtype=np.float64)
    
    for scale in [3, 5, 7, 9]:
        for angle in np.linspace(0, np.pi, 6, endpoint=False):
            kernel = cv2.getGaborKernel((21, 21), scale, angle, 10, 0.5, 0)
            filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
            edges += np.abs(filtered)
    
    # Normalize
    edges = edges / edges.max()
    
    # Combine with gradient-based edges
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = grad_mag / (grad_mag.max() + 1e-6)
    
    # Weighted combination
    combined = 0.7 * edges + 0.3 * grad_mag
    
    return combined

def find_robust_center_enhanced(edge_points, image_shape, num_pairs=10000):
    """
    Enhanced perpendicular bisector voting from concen_split4.py
    with additional stability checks.
    """
    h, w = image_shape
    
    # Multi-resolution voting for robustness
    resolutions = [100, 200, 400]
    all_centers = []
    
    for resolution in resolutions:
        accumulator = np.zeros((resolution, resolution))
        
        n_points = len(edge_points)
        if n_points < 50:
            continue
            
        # Sample pairs
        indices1 = np.random.choice(n_points, num_pairs, replace=True)
        indices2 = np.random.choice(n_points, num_pairs, replace=True)
        
        # Remove identical pairs
        mask = indices1 != indices2
        indices1 = indices1[mask]
        indices2 = indices2[mask]
        
        p1s = edge_points[indices1]
        p2s = edge_points[indices2]
        
        # Calculate perpendicular bisectors
        midpoints = (p1s + p2s) / 2
        vectors = p2s - p1s
        
        # Filter out very short segments
        lengths = np.linalg.norm(vectors, axis=1)
        mask = lengths > 10
        midpoints = midpoints[mask]
        vectors = vectors[mask]
        
        # Perpendicular direction
        perp_vectors = np.column_stack([-vectors[:, 1], vectors[:, 0]])
        perp_vectors = perp_vectors / (np.linalg.norm(perp_vectors, axis=1, keepdims=True) + 1e-6)
        
        # Vote along perpendicular lines
        for midpoint, perp_vec in zip(midpoints, perp_vectors):
            # Sample points along the perpendicular line
            t_values = np.linspace(-max(h, w), max(h, w), 1000)
            line_points = midpoint + t_values[:, np.newaxis] * perp_vec
            
            # Filter points within image bounds
            mask = (line_points[:, 0] >= 0) & (line_points[:, 0] < w) & \
                   (line_points[:, 1] >= 0) & (line_points[:, 1] < h)
            line_points = line_points[mask]
            
            if len(line_points) > 0:
                # Convert to accumulator indices
                x_idx = (line_points[:, 0] / w * resolution).astype(int)
                y_idx = (line_points[:, 1] / h * resolution).astype(int)
                
                # Increment accumulator
                np.add.at(accumulator, (y_idx, x_idx), 1)
        
        # Find peak with Gaussian smoothing
        accumulator = ndimage.gaussian_filter(accumulator, sigma=2)
        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        
        center_x = max_idx[1] * w / resolution
        center_y = max_idx[0] * h / resolution
        
        all_centers.append((center_x, center_y))
    
    # Robust averaging of multi-resolution results
    if all_centers:
        centers_array = np.array(all_centers)
        # Use median for robustness
        final_center = np.median(centers_array, axis=0)
        return final_center[0], final_center[1]
    else:
        return w / 2, h / 2

def detect_radii_multi_method(edge_points, center):
    """
    Multi-method radius detection combining histogram and Fourier analysis.
    """
    center_arr = np.array(center)
    distances = np.linalg.norm(edge_points - center_arr, axis=1)
    
    # Method 1: Histogram analysis with adaptive binning
    n_bins = min(200, len(distances) // 10)
    hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, np.max(distances)))
    
    # Smooth histogram
    hist_smooth = ndimage.gaussian_filter1d(hist.astype(float), sigma=2)
    
    # Find peaks
    peaks, properties = signal.find_peaks(hist_smooth, 
                                         prominence=np.max(hist_smooth) * 0.1,
                                         distance=5)
    
    # Get two most prominent peaks
    if len(peaks) >= 2:
        # Sort by prominence
        prominences = properties['prominences']
        sorted_idx = np.argsort(prominences)[-2:]
        peak_indices = peaks[sorted_idx]
        
        r1_hist = bin_edges[peak_indices[0]] + (bin_edges[1] - bin_edges[0]) / 2
        r2_hist = bin_edges[peak_indices[1]] + (bin_edges[1] - bin_edges[0]) / 2
    else:
        # Fallback
        r1_hist = np.percentile(distances, 25)
        r2_hist = np.percentile(distances, 75)
    
    # Method 2: Fourier analysis of radial profile
    max_r = int(np.max(distances))
    radial_profile = np.zeros(max_r)
    counts = np.zeros(max_r)
    
    for d in distances:
        r = int(d)
        if r < max_r:
            radial_profile[r] += 1
            counts[r] += 1
    
    # Normalize
    mask = counts > 0
    radial_profile[mask] /= counts[mask]
    
    # FFT to find periodic structures
    radial_fft = np.abs(fft(radial_profile))
    freqs = fftfreq(len(radial_profile))
    
    # Find peaks in frequency domain
    fft_peaks, _ = signal.find_peaks(radial_fft[1:len(radial_fft)//2], 
                                     height=np.max(radial_fft) * 0.1)
    
    # Combine both methods with weighted average
    r1 = min(r1_hist, r2_hist)
    r2 = max(r1_hist, r2_hist)
    
    return r1, r2

def refine_with_constraints(edge_points, initial_params):
    """
    Advanced refinement with geometric constraints and robust loss.
    Combines best practices from multiple scripts.
    """
    def residuals(params, points):
        cx, cy, r1, r2 = params
        center = np.array([cx, cy])
        distances = np.linalg.norm(points - center, axis=1)
        
        # Distance to nearest circle
        err1 = np.abs(distances - r1)
        err2 = np.abs(distances - r2)
        errors = np.minimum(err1, err2)
        
        # Add geometric constraints as soft penalties
        # Ensure r2 > r1
        if r2 <= r1:
            errors = np.append(errors, 100 * (r1 - r2 + 1))
        
        # Ensure reasonable ratio
        ratio = r1 / r2 if r2 > 0 else 0
        if ratio < 0.1 or ratio > 0.9:
            errors = np.append(errors, 50 * abs(ratio - 0.5))
        
        return errors
    
    # Set bounds based on image statistics
    cx_init, cy_init, r1_init, r2_init = initial_params
    
    bounds = ([cx_init - 50, cy_init - 50, 5, 10],
              [cx_init + 50, cy_init + 50, r2_init * 0.8, r2_init * 1.5])
    
    # Use Trust Region Reflective with Cauchy loss for robustness
    result = least_squares(residuals, initial_params, 
                          args=(edge_points,),
                          bounds=bounds,
                          method='trf',
                          loss='cauchy',
                          f_scale=1.0)
    
    final_params = result.x
    
    # Ensure correct ordering
    if final_params[2] > final_params[3]:
        final_params[2], final_params[3] = final_params[3], final_params[2]
    
    return final_params

def quality_check(edge_points, params, threshold=0.3):
    """
    Quality check to ensure the detection is reasonable.
    """
    cx, cy, r1, r2 = params
    center = np.array([cx, cy])
    distances = np.linalg.norm(edge_points - center, axis=1)
    
    # Count inliers
    err1 = np.abs(distances - r1)
    err2 = np.abs(distances - r2)
    
    inlier_threshold = 3.0
    inliers1 = np.sum(err1 < inlier_threshold)
    inliers2 = np.sum(err2 < inlier_threshold)
    
    total_inliers = inliers1 + inliers2
    inlier_ratio = total_inliers / len(edge_points)
    
    return inlier_ratio > threshold

def create_final_masks(image, params):
    """
    Create high-quality masks with anti-aliasing.
    """
    h, w = image.shape
    cx, cy, r_core, r_cladding = params
    
    # Create high-resolution masks for anti-aliasing
    scale = 4
    h_hr, w_hr = h * scale, w * scale
    
    y, x = np.ogrid[:h_hr, :w_hr]
    cx_hr, cy_hr = cx * scale, cy * scale
    r_core_hr = r_core * scale
    r_cladding_hr = r_cladding * scale
    
    dist_sq = (x - cx_hr)**2 + (y - cy_hr)**2
    
    # Create masks
    core_mask_hr = (dist_sq <= r_core_hr**2).astype(np.float32)
    cladding_mask_hr = ((dist_sq > r_core_hr**2) & 
                        (dist_sq <= r_cladding_hr**2)).astype(np.float32)
    
    # Apply Gaussian smoothing for anti-aliasing
    core_mask_hr = ndimage.gaussian_filter(core_mask_hr, sigma=scale/2)
    cladding_mask_hr = ndimage.gaussian_filter(cladding_mask_hr, sigma=scale/2)
    
    # Downsample
    core_mask = cv2.resize(core_mask_hr, (w, h), interpolation=cv2.INTER_AREA)
    cladding_mask = cv2.resize(cladding_mask_hr, (w, h), interpolation=cv2.INTER_AREA)
    
    # Apply masks
    isolated_core = (image * core_mask).astype(np.uint8)
    isolated_cladding = (image * cladding_mask).astype(np.uint8)
    
    # Crop to content
    def crop_to_content(img, mask):
        coords = np.argwhere(mask > 0.5)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            return img[y_min:y_max+1, x_min:x_max+1]
        return img
    
    isolated_core = crop_to_content(isolated_core, core_mask)
    isolated_cladding = crop_to_content(isolated_cladding, cladding_mask)
    
    return isolated_core, isolated_cladding

def unified_fiber_splitter(image_path, output_dir='unified_output'):
    """
    Main processing pipeline combining all best practices.
    """
    print(f"\nProcessing: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Stage 1: Advanced preprocessing
    print("Stage 1: Preprocessing...")
    denoised = wavelet_denoise(image)
    preprocessed = cv2.medianBlur(denoised, 5)
    
    # Enhance contrast if needed
    if np.std(preprocessed) < 30:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        preprocessed = clahe.apply(preprocessed)
    
    # Stage 2: Multi-method edge detection
    print("Stage 2: Edge detection...")
    # Phase congruency edges
    pc_edges = phase_congruency_edges(preprocessed)
    
    # Standard Canny as backup
    canny_edges = cv2.Canny(preprocessed, 50, 150)
    
    # Combine edges
    combined_edges = np.maximum(pc_edges, canny_edges / 255.0)
    edge_binary = (combined_edges > 0.1).astype(np.uint8) * 255
    
    # Extract edge points
    edge_points = np.argwhere(edge_binary > 0).astype(float)[:, ::-1]
    
    if len(edge_points) < 100:
        print("Error: Insufficient edge points detected")
        return None
    
    print(f"  Detected {len(edge_points)} edge points")
    
    # Stage 3: Robust center detection
    print("Stage 3: Center detection...")
    center = find_robust_center_enhanced(edge_points, image.shape)
    print(f"  Center: ({center[0]:.2f}, {center[1]:.2f})")
    
    # Stage 4: Multi-method radius detection
    print("Stage 4: Radius detection...")
    r1, r2 = detect_radii_multi_method(edge_points, center)
    
    initial_params = [center[0], center[1], r1, r2]
    print(f"  Initial radii: {r1:.2f}, {r2:.2f}")
    
    # Stage 5: Refinement with constraints
    print("Stage 5: Parameter refinement...")
    final_params = refine_with_constraints(edge_points, initial_params)
    
    # Quality check
    if not quality_check(edge_points, final_params):
        print("  Warning: Low quality detection, trying alternative approach...")
        # Fallback to simpler method
        circles = cv2.HoughCircles(preprocessed, cv2.HOUGH_GRADIENT,
                                  dp=1, minDist=50,
                                  param1=50, param2=30,
                                  minRadius=20, maxRadius=min(image.shape)//2)
        
        if circles is not None and len(circles[0]) >= 2:
            circles = circles[0][:2]
            cx = np.mean(circles[:, 0])
            cy = np.mean(circles[:, 1])
            radii = sorted(circles[:, 2])
            final_params = [cx, cy, radii[0], radii[1]]
    
    cx, cy, r_core, r_cladding = final_params
    print(f"  Final: Center=({cx:.2f}, {cy:.2f}), Core={r_core:.2f}, Cladding={r_cladding:.2f}")
    
    # Stage 6: Create and save outputs
    print("Stage 6: Creating outputs...")
    isolated_core, isolated_cladding = create_final_masks(image, final_params)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_core.png"), isolated_core)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_cladding.png"), isolated_cladding)
    
    print(f"Successfully saved results to '{output_dir}'")
    
    return {
        'center': (cx, cy),
        'core_radius': r_core,
        'cladding_radius': r_cladding
    }

if __name__ == '__main__':
    # Process the test image
    test_image = r'C:\Users\Saem1001\Documents\GitHub\IPPS\App\images\img63.jpg'
    unified_fiber_splitter(test_image)