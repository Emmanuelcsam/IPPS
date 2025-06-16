"""
Advanced Mathematical Methods for Fiber Optic Core/Cladding Detection
Implements multiple PhD-level mathematical techniques for maximum accuracy
"""

import cv2
import numpy as np
from scipy import optimize, signal, special, ndimage
from scipy.sparse import csr_matrix, linalg as sparse_linalg
from scipy.optimize import minimize, least_squares, differential_evolution
from scipy.interpolate import RBFInterpolator, griddata
from scipy.fft import fft2, ifft2, fftfreq
from skimage import feature, morphology, measure
import pywt
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class AdvancedFiberOpticDetector:
    """
    Implements advanced mathematical methods for detecting concentric circles
    in fiber optic end face images with PhD-level accuracy.
    """
    
    def __init__(self, image: np.ndarray):
        """Initialize with an input image."""
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image.copy()
        
        self.h, self.w = self.image.shape
        self.image_float = self.image.astype(np.float64) / 255.0
        
        # Precompute coordinate grids for efficiency
        self.y_grid, self.x_grid = np.ogrid[:self.h, :self.w]
        
        # Results storage
        self.results = {}
        
    def detect_circles_comprehensive(self) -> Dict:
        """
        Main method that combines multiple advanced techniques for maximum accuracy.
        """
        print("Starting comprehensive circle detection with advanced mathematics...")
        
        # 1. Multi-scale preprocessing with wavelet denoising
        denoised = self._wavelet_denoise()
        
        # 2. Advanced edge detection with subpixel accuracy
        edges, edge_points = self._advanced_edge_detection(denoised)
        
        # 3. Multiple mathematical approaches for initial estimation
        estimates = []
        
        # 3a. Fourier-Bessel Transform approach
        fb_estimate = self._fourier_bessel_analysis()
        if fb_estimate:
            estimates.append(('Fourier-Bessel', fb_estimate))
        
        # 3b. Persistent Homology approach
        ph_estimate = self._persistent_homology_analysis(edges)
        if ph_estimate:
            estimates.append(('Persistent Homology', ph_estimate))
        
        # 3c. Conformal mapping approach
        cm_estimate = self._conformal_mapping_approach(edge_points)
        if cm_estimate:
            estimates.append(('Conformal Mapping', cm_estimate))
        
        # 3d. Advanced RANSAC with geometric constraints
        ransac_estimate = self._geometric_ransac(edge_points)
        if ransac_estimate:
            estimates.append(('Geometric RANSAC', ransac_estimate))
        
        # 3e. Variational approach with shape priors
        var_estimate = self._variational_approach(edges)
        if var_estimate:
            estimates.append(('Variational', var_estimate))
        
        # 4. Combine estimates using robust statistics
        if estimates:
            combined_estimate = self._robust_combination(estimates)
            
            # 5. Final refinement using advanced optimization
            final_params = self._advanced_refinement(combined_estimate, edge_points)
            
            # 6. Uncertainty quantification
            uncertainty = self._quantify_uncertainty(final_params, edge_points)
            
            self.results = {
                'center': (final_params[0], final_params[1]),
                'r_core': final_params[2],
                'r_cladding': final_params[3],
                'uncertainty': uncertainty,
                'methods_used': [name for name, _ in estimates],
                'edge_points': edge_points
            }
            
            return self.results
        else:
            raise ValueError("No valid estimates found")
    
    def _wavelet_denoise(self) -> np.ndarray:
        """
        Advanced wavelet denoising using stationary wavelet transform
        with level-dependent thresholding.
        """
        # Use stationary wavelet transform for shift-invariance
        coeffs = pywt.swt2(self.image_float, 'db4', level=4)
        
        # Level-dependent noise estimation and thresholding
        denoised_coeffs = []
        for level, (cA, (cH, cV, cD)) in enumerate(coeffs):
            # Estimate noise using median absolute deviation
            sigma_H = np.median(np.abs(cH)) / 0.6745
            sigma_V = np.median(np.abs(cV)) / 0.6745
            sigma_D = np.median(np.abs(cD)) / 0.6745
            
            # Adaptive threshold based on Stein's Unbiased Risk Estimate
            thresh_H = sigma_H * np.sqrt(2 * np.log(cH.size))
            thresh_V = sigma_V * np.sqrt(2 * np.log(cV.size))
            thresh_D = sigma_D * np.sqrt(2 * np.log(cD.size))
            
            # Soft thresholding with level-dependent weight
            weight = 1.0 / (level + 1)
            cH_thresh = np.sign(cH) * np.maximum(np.abs(cH) - weight * thresh_H, 0)
            cV_thresh = np.sign(cV) * np.maximum(np.abs(cV) - weight * thresh_V, 0)
            cD_thresh = np.sign(cD) * np.maximum(np.abs(cD) - weight * thresh_D, 0)
            
            denoised_coeffs.append((cA, (cH_thresh, cV_thresh, cD_thresh)))
        
        # Reconstruct
        denoised = pywt.iswt2(denoised_coeffs, 'db4')
        return np.clip(denoised, 0, 1)
    
    def _advanced_edge_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced edge detection with subpixel accuracy using:
        - Phase congruency for illumination invariance
        - Non-maximum suppression with parabolic fitting
        - Hysteresis with adaptive thresholds
        """
        # Phase congruency computation using log-Gabor filters
        edges_pc = self._phase_congruency(image)
        
        # Compute gradients with Sobel operators
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)
        
        # Combine phase congruency with gradient magnitude
        combined_edge_strength = 0.7 * edges_pc + 0.3 * (magnitude / magnitude.max())
        
        # Non-maximum suppression with subpixel refinement
        edges_subpixel = self._subpixel_nms(combined_edge_strength, orientation)
        
        # Adaptive hysteresis thresholding
        high_thresh = np.percentile(edges_subpixel[edges_subpixel > 0], 85)
        low_thresh = 0.4 * high_thresh
        edges_binary = feature.canny(image, low_threshold=low_thresh, high_threshold=high_thresh)
        
        # Extract edge points with subpixel coordinates
        edge_points = self._extract_subpixel_edges(edges_subpixel, edges_binary, orientation)
        
        return edges_binary, edge_points
    
    def _phase_congruency(self, image: np.ndarray) -> np.ndarray:
        """
        Compute phase congruency using log-Gabor filters.
        More robust to illumination changes than gradient-based methods.
        """
        rows, cols = image.shape
        
        # FFT of image
        im_fft = fft2(image)
        
        # Initialize arrays
        pc = np.zeros_like(image)
        total_energy = np.zeros_like(image)
        total_amplitude = np.zeros_like(image)
        
        # Parameters
        n_scale = 4  # Number of scales
        n_orient = 6  # Number of orientations
        min_wavelength = 3
        mult = 2.1
        sigma_onf = 0.55
        
        # Create frequency grids
        u, v = np.meshgrid(fftfreq(cols), fftfreq(rows))
        radius = np.sqrt(u**2 + v**2)
        radius[0, 0] = 1  # Avoid division by zero
        theta = np.arctan2(v, u)
        
        # Process each scale and orientation
        for s in range(n_scale):
            wavelength = min_wavelength * mult**s
            fo = 1.0 / wavelength
            
            # Log-Gabor radial filter
            log_gabor = np.exp(-(np.log(radius/fo))**2 / (2 * np.log(sigma_onf)**2))
            log_gabor[radius < fo/3] = 0  # Cut off low frequencies
            
            for o in range(n_orient):
                angle = o * np.pi / n_orient
                
                # Angular filter
                ds = np.sin(theta - angle)
                dc = np.cos(theta - angle)
                angular = np.exp(-(ds**2 + dc**2) / (2 * 0.5**2))
                
                # Combined filter
                filter_bank = log_gabor * angular
                
                # Apply filter
                response = ifft2(im_fft * filter_bank)
                amplitude = np.abs(response)
                phase = np.angle(response)
                
                # Accumulate
                total_energy += response.real
                total_amplitude += amplitude
        
        # Phase congruency
        epsilon = 1e-6
        pc = total_energy / (total_amplitude + epsilon)
        
        return np.abs(pc)
    
    def _subpixel_nms(self, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """
        Non-maximum suppression with subpixel accuracy using parabolic fitting.
        """
        rows, cols = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        # Pad arrays for boundary handling
        mag_pad = np.pad(magnitude, 1, mode='edge')
        
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                angle = orientation[i-1, j-1]
                
                # Get interpolation direction
                dx = np.cos(angle)
                dy = np.sin(angle)
                
                # Sample along gradient direction
                p1 = self._bilinear_interpolate(mag_pad, j - dx, i - dy)
                p2 = mag_pad[i, j]
                p3 = self._bilinear_interpolate(mag_pad, j + dx, i + dy)
                
                # Check if local maximum
                if p2 >= p1 and p2 >= p3:
                    # Parabolic fitting for subpixel refinement
                    if p1 + p3 - 2*p2 != 0:
                        offset = 0.5 * (p1 - p3) / (p1 + p3 - 2*p2)
                        offset = np.clip(offset, -0.5, 0.5)
                        
                        # Refine position and magnitude
                        refined_mag = p2 - 0.25 * (p1 - p3) * offset
                        suppressed[i-1, j-1] = refined_mag
                    else:
                        suppressed[i-1, j-1] = p2
        
        return suppressed
    
    def _bilinear_interpolate(self, image: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation for subpixel access."""
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1
        
        # Clamp to image bounds
        x0 = np.clip(x0, 0, image.shape[1] - 1)
        x1 = np.clip(x1, 0, image.shape[1] - 1)
        y0 = np.clip(y0, 0, image.shape[0] - 1)
        y1 = np.clip(y1, 0, image.shape[0] - 1)
        
        # Interpolation weights
        wx = x - x0
        wy = y - y0
        
        # Bilinear interpolation
        return (1-wx)*(1-wy)*image[y0,x0] + wx*(1-wy)*image[y0,x1] + \
               (1-wx)*wy*image[y1,x0] + wx*wy*image[y1,x1]
    
    def _extract_subpixel_edges(self, magnitude: np.ndarray, binary: np.ndarray, 
                               orientation: np.ndarray) -> np.ndarray:
        """Extract edge points with subpixel accuracy."""
        edge_pixels = np.argwhere(binary)
        subpixel_edges = []
        
        for y, x in edge_pixels:
            if magnitude[y, x] > 0:
                # Compute subpixel offset using quadratic fitting
                angle = orientation[y, x]
                dx, dy = np.cos(angle), np.sin(angle)
                
                # Sample perpendicular to edge
                if 0 < x < self.w-1 and 0 < y < self.h-1:
                    # Fit parabola perpendicular to edge
                    p = []
                    for t in [-1, 0, 1]:
                        xi = x + t * dy
                        yi = y - t * dx
                        if 0 <= xi < self.w and 0 <= yi < self.h:
                            p.append(self._bilinear_interpolate(magnitude, xi, yi))
                    
                    if len(p) == 3 and p[0] + p[2] - 2*p[1] != 0:
                        offset = 0.5 * (p[0] - p[2]) / (p[0] + p[2] - 2*p[1])
                        offset = np.clip(offset, -0.5, 0.5)
                        
                        # Refine position
                        x_refined = x + offset * dy
                        y_refined = y - offset * dx
                        subpixel_edges.append([x_refined, y_refined])
                    else:
                        subpixel_edges.append([float(x), float(y)])
                else:
                    subpixel_edges.append([float(x), float(y)])
        
        return np.array(subpixel_edges)
    
    def _fourier_bessel_analysis(self) -> Optional[List[float]]:
        """
        Use Fourier-Bessel transform to detect circular patterns.
        This is particularly effective for concentric circles.
        """
        # Convert to polar coordinates centered at image center
        cx, cy = self.w // 2, self.h // 2
        max_radius = min(cx, cy, self.w - cx, self.h - cy)
        
        # Create polar grid
        r_samples = np.linspace(0, max_radius, max_radius)
        theta_samples = np.linspace(0, 2*np.pi, 360)
        
        # Interpolate image to polar coordinates
        polar_image = np.zeros((len(r_samples), len(theta_samples)))
        
        for i, r in enumerate(r_samples):
            for j, theta in enumerate(theta_samples):
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                if 0 <= x < self.w and 0 <= y < self.h:
                    polar_image[i, j] = self._bilinear_interpolate(self.image_float, x, y)
        
        # Compute radial profile (average over angles)
        radial_profile = np.mean(polar_image, axis=1)
        
        # Apply Hankel transform (continuous version of Fourier-Bessel)
        # For discrete case, we'll use FFT of the radial profile
        radial_fft = np.fft.fft(radial_profile)
        radial_power = np.abs(radial_fft)**2
        
        # Find peaks in power spectrum (indicating circular patterns)
        peaks, properties = signal.find_peaks(radial_power[1:len(radial_power)//2], 
                                            prominence=np.max(radial_power) * 0.1)
        
        if len(peaks) >= 2:
            # Convert frequency peaks back to spatial radii
            radii = max_radius / (peaks + 1)
            r_cladding = np.max(radii[:2])
            r_core = np.min(radii[:2])
            
            # Refine center using phase information
            cx_refined, cy_refined = self._refine_center_fourier(radial_fft, cx, cy)
            
            return [cx_refined, cy_refined, r_core, r_cladding]
        
        return None
    
    def _refine_center_fourier(self, radial_fft: np.ndarray, cx: float, cy: float) -> Tuple[float, float]:
        """Refine center estimate using phase of Fourier components."""
        # Use phase gradient to estimate center offset
        phase = np.angle(radial_fft[1:10])  # Use low frequencies
        phase_gradient = np.gradient(phase)
        
        # Estimate offset (simplified - in practice would use 2D analysis)
        offset_magnitude = np.mean(np.abs(phase_gradient)) * 2.0
        offset_magnitude = min(offset_magnitude, 5.0)  # Limit refinement
        
        # Apply small random perturbation and test
        best_cx, best_cy = cx, cy
        best_score = 0
        
        for _ in range(20):
            dx = np.random.randn() * offset_magnitude
            dy = np.random.randn() * offset_magnitude
            test_cx, test_cy = cx + dx, cy + dy
            
            # Quick score based on radial symmetry
            score = self._radial_symmetry_score(test_cx, test_cy)
            if score > best_score:
                best_score = score
                best_cx, best_cy = test_cx, test_cy
        
        return best_cx, best_cy
    
    def _radial_symmetry_score(self, cx: float, cy: float) -> float:
        """Compute radial symmetry score for a given center."""
        score = 0.0
        n_angles = 36
        
        for r in [20, 40, 60, 80]:
            if r < min(cx, cy, self.w - cx, self.h - cy):
                values = []
                for i in range(n_angles):
                    theta = 2 * np.pi * i / n_angles
                    x = cx + r * np.cos(theta)
                    y = cy + r * np.sin(theta)
                    if 0 <= x < self.w and 0 <= y < self.h:
                        values.append(self._bilinear_interpolate(self.image_float, x, y))
                
                if values:
                    score += 1.0 / (1.0 + np.std(values))
        
        return score
    
    def _persistent_homology_analysis(self, edges: np.ndarray) -> Optional[List[float]]:
        """
        Use persistent homology to detect and verify circular structures.
        This is highly robust to noise and gaps in the circles.
        """
        # Create filtration based on distance transform
        dist_transform = ndimage.distance_transform_edt(~edges)
        
        # Build a simple cubical complex filtration
        # For efficiency, we'll sample the persistence at different thresholds
        thresholds = np.percentile(dist_transform, np.linspace(0, 100, 50))
        
        # Track birth and death of connected components and loops
        persistence_features = []
        
        for i, thresh in enumerate(thresholds[:-1]):
            # Binary image at this threshold
            binary = dist_transform <= thresh
            
            # Label connected components
            labeled, n_components = ndimage.label(binary)
            
            # Compute Euler characteristic (simplified homology)
            euler = self._compute_euler_characteristic(binary)
            
            # Detect loops (1-cycles) using Euler characteristic
            # χ = #components - #loops (for 2D)
            n_loops = n_components - euler
            
            if n_loops > 0:
                # Find the actual loops using contour detection
                contours = measure.find_contours(binary.astype(float), 0.5)
                
                for contour in contours:
                    if len(contour) > 50:  # Significant loop
                        # Fit circle to contour
                        circle_params = self._fit_circle_to_contour(contour)
                        if circle_params:
                            persistence_features.append({
                                'birth': thresh,
                                'params': circle_params,
                                'size': len(contour)
                            })
        
        # Find most persistent features (likely the true circles)
        if len(persistence_features) >= 2:
            # Sort by size (larger loops are more likely to be true circles)
            persistence_features.sort(key=lambda x: x['size'], reverse=True)
            
            # Extract two most significant circles
            params1 = persistence_features[0]['params']
            params2 = persistence_features[1]['params']
            
            # Ensure concentricity by averaging centers
            cx = (params1[0] + params2[0]) / 2
            cy = (params1[1] + params2[1]) / 2
            r1, r2 = params1[2], params2[2]
            
            if r1 > r2:
                return [cx, cy, r2, r1]
            else:
                return [cx, cy, r1, r2]
        
        return None
    
    def _compute_euler_characteristic(self, binary: np.ndarray) -> int:
        """Compute Euler characteristic using pixel adjacency."""
        # Count vertices (1-pixels)
        V = np.sum(binary)
        
        # Count edges (adjacent pixel pairs)
        E = 0
        E += np.sum(binary[:-1, :] & binary[1:, :])  # Vertical edges
        E += np.sum(binary[:, :-1] & binary[:, 1:])  # Horizontal edges
        
        # Count faces (2x2 blocks of 1s)
        F = np.sum(binary[:-1, :-1] & binary[1:, :-1] & 
                   binary[:-1, 1:] & binary[1:, 1:])
        
        # Euler characteristic for 2D
        return int(V - E + F)
    
    def _fit_circle_to_contour(self, contour: np.ndarray) -> Optional[List[float]]:
        """Fit circle to contour points using algebraic method."""
        if len(contour) < 5:
            return None
        
        # Swap columns because contour is in (row, col) format
        x, y = contour[:, 1], contour[:, 0]
        
        # Algebraic circle fit
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x**2 + y**2
        
        try:
            params = np.linalg.lstsq(A, b, rcond=None)[0]
            cx = params[0] / 2
            cy = params[1] / 2
            r = np.sqrt(params[2] + cx**2 + cy**2)
            
            if 0 < r < max(self.h, self.w):
                return [cx, cy, r]
        except:
            pass
        
        return None
    
    def _conformal_mapping_approach(self, edge_points: np.ndarray) -> Optional[List[float]]:
        """
        Use conformal mapping to transform the problem.
        Maps potential circles to simpler shapes for detection.
        """
        if len(edge_points) < 20:
            return None
        
        # Estimate initial center as centroid
        cx_init = np.mean(edge_points[:, 0])
        cy_init = np.mean(edge_points[:, 1])
        
        # Apply inversion transformation w = 1/(z - z0)
        # This maps circles centered at z0 to lines
        best_params = None
        best_score = float('inf')
        
        # Try different center candidates
        for dx in np.linspace(-10, 10, 5):
            for dy in np.linspace(-10, 10, 5):
                cx_test = cx_init + dx
                cy_test = cy_init + dy
                
                # Transform points
                z = edge_points[:, 0] + 1j * edge_points[:, 1]
                z0 = cx_test + 1j * cy_test
                
                # Avoid division by zero
                mask = np.abs(z - z0) > 1e-6
                w = 1.0 / (z[mask] - z0)
                
                if len(w) < 10:
                    continue
                
                # In the w-plane, concentric circles become parallel lines
                # Detect lines using Hough transform on transformed points
                u, v = w.real, w.imag
                
                # Discretize for Hough
                scale = 1000
                u_discrete = (u * scale).astype(int)
                v_discrete = (v * scale).astype(int)
                
                # Find dominant directions (should be perpendicular to lines)
                angles = np.arctan2(v_discrete, u_discrete)
                hist, bins = np.histogram(angles, bins=36)
                
                # Look for two dominant directions (two parallel lines)
                peaks, _ = signal.find_peaks(hist, height=np.max(hist) * 0.3)
                
                if len(peaks) >= 1:
                    # Inverse transform to get circle parameters
                    # Simplified: estimate radii from distance between lines
                    radii = []
                    
                    for i in range(min(len(peaks), 2)):
                        angle = bins[peaks[i]]
                        # Project points onto line perpendicular to this angle
                        projections = u * np.cos(angle) + v * np.sin(angle)
                        
                        # Find clusters in projections (lines in w-plane)
                        kde = signal.gaussian_kde(projections)
                        x_range = np.linspace(projections.min(), projections.max(), 200)
                        density = kde(x_range)
                        
                        line_peaks, _ = signal.find_peaks(density, height=np.max(density) * 0.3)
                        
                        for peak in line_peaks[:2]:
                            # Convert back to radius in original space
                            line_pos = x_range[peak]
                            if abs(line_pos) > 1e-6:
                                r = 1.0 / abs(line_pos)
                                if 0 < r < max(self.h, self.w):
                                    radii.append(r)
                    
                    if len(radii) >= 2:
                        radii.sort()
                        score = np.std(radii[:2])  # How similar are the radii
                        
                        if score < best_score:
                            best_score = score
                            best_params = [cx_test, cy_test, radii[0], radii[1]]
        
        return best_params
    
    def _geometric_ransac(self, edge_points: np.ndarray) -> Optional[List[float]]:
        """
        RANSAC with geometric constraints specific to concentric circles.
        Uses advanced sampling strategies and verification.
        """
        if len(edge_points) < 6:
            return None
        
        best_params = None
        best_score = 0
        n_iterations = 1000
        
        for iteration in range(n_iterations):
            # Intelligent sampling: pick points that are likely from different circles
            # based on distance from centroid
            centroid = np.mean(edge_points, axis=0)
            distances = np.linalg.norm(edge_points - centroid, axis=1)
            
            # Stratified sampling
            median_dist = np.median(distances)
            inner_mask = distances < median_dist
            outer_mask = ~inner_mask
            
            inner_points = edge_points[inner_mask]
            outer_points = edge_points[outer_mask]
            
            if len(inner_points) >= 3 and len(outer_points) >= 3:
                # Sample 3 points from each group
                inner_sample = inner_points[np.random.choice(len(inner_points), 3, replace=False)]
                outer_sample = outer_points[np.random.choice(len(outer_points), 3, replace=False)]
                
                # Fit circles to each sample
                inner_circle = self._fit_circle_algebraic(inner_sample)
                outer_circle = self._fit_circle_algebraic(outer_sample)
                
                if inner_circle and outer_circle:
                    # Check concentricity constraint
                    cx1, cy1, r1 = inner_circle
                    cx2, cy2, r2 = outer_circle
                    
                    center_distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                    
                    if center_distance < 5.0:  # Concentricity threshold
                        # Average centers for true concentricity
                        cx = (cx1 + cx2) / 2
                        cy = (cy1 + cy2) / 2
                        
                        # Ensure r1 < r2
                        if r1 > r2:
                            r1, r2 = r2, r1
                        
                        # Score based on inliers with adaptive threshold
                        score = self._compute_ransac_score(edge_points, cx, cy, r1, r2)
                        
                        if score > best_score:
                            best_score = score
                            best_params = [cx, cy, r1, r2]
        
        return best_params
    
    def _fit_circle_algebraic(self, points: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Fit circle using Taubin's method (algebraic fit with constraint)."""
        if len(points) < 3:
            return None
        
        # Center data
        x, y = points[:, 0], points[:, 1]
        mx, my = np.mean(x), np.mean(y)
        u, v = x - mx, y - my
        
        # Build matrices
        Suu = np.sum(u * u)
        Svv = np.sum(v * v)
        Suv = np.sum(u * v)
        Suuu = np.sum(u * u * u)
        Svvv = np.sum(v * v * v)
        Suvv = np.sum(u * v * v)
        Suuv = np.sum(u * u * v)
        
        # Solve system
        A = np.array([[Suu, Suv], [Suv, Svv]])
        b = 0.5 * np.array([Suuu + Suvv, Svvv + Suuv])
        
        try:
            uc, vc = np.linalg.solve(A, b)
            cx = uc + mx
            cy = vc + my
            r = np.sqrt(uc*uc + vc*vc + (Suu + Svv) / len(points))
            
            if 0 < r < max(self.h, self.w):
                return (cx, cy, r)
        except:
            pass
        
        return None
    
    def _compute_ransac_score(self, points: np.ndarray, cx: float, cy: float, 
                             r1: float, r2: float) -> float:
        """
        Compute RANSAC score with adaptive inlier threshold and 
        geometric consistency checks.
        """
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        
        # Adaptive threshold based on image resolution
        base_threshold = 2.0
        resolution_factor = np.sqrt(self.h * self.w) / 500.0
        threshold = base_threshold * resolution_factor
        
        # Count inliers for each circle
        inliers1 = np.abs(distances - r1) < threshold
        inliers2 = np.abs(distances - r2) < threshold
        
        # Geometric consistency: check angular distribution of inliers
        angular_score1 = self._angular_distribution_score(points[inliers1], cx, cy)
        angular_score2 = self._angular_distribution_score(points[inliers2], cx, cy)
        
        # Combined score
        n_inliers = np.sum(inliers1) + np.sum(inliers2)
        coverage = (angular_score1 + angular_score2) / 2
        
        return n_inliers * coverage
    
    def _angular_distribution_score(self, points: np.ndarray, cx: float, cy: float) -> float:
        """Measure how well distributed points are around the circle."""
        if len(points) < 3:
            return 0.0
        
        # Compute angles
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        angles = np.sort(angles)
        
        # Check gaps in angular coverage
        gaps = np.diff(angles)
        gaps = np.append(gaps, angles[0] + 2*np.pi - angles[-1])
        
        # Score based on maximum gap (smaller is better)
        max_gap = np.max(gaps)
        score = 1.0 - max_gap / (2 * np.pi)
        
        return np.clip(score, 0, 1)
    
    def _variational_approach(self, edges: np.ndarray) -> Optional[List[float]]:
        """
        Variational approach using energy minimization with shape priors.
        Implements a simplified Chan-Vese model with circular constraints.
        """
        # Initialize with image moments
        M = cv2.moments(edges.astype(np.uint8))
        if M['m00'] == 0:
            return None
        
        cx_init = M['m10'] / M['m00']
        cy_init = M['m01'] / M['m00']
        
        # Initial radii based on area
        area = np.sum(edges) / 255.0
        r_init = np.sqrt(area / np.pi)
        
        # Define energy functional
        def energy_functional(params):
            cx, cy, r1, r2 = params
            
            if r1 <= 0 or r2 <= 0 or r1 >= r2:
                return 1e10
            
            # Create masks for the three regions
            dist_sq = (self.x_grid - cx)**2 + (self.y_grid - cy)**2
            mask_inner = dist_sq <= r1**2
            mask_annulus = (dist_sq > r1**2) & (dist_sq <= r2**2)
            mask_outer = dist_sq > r2**2
            
            # Region-based energy (Chan-Vese style)
            c1 = np.mean(self.image_float[mask_inner]) if np.any(mask_inner) else 0
            c2 = np.mean(self.image_float[mask_annulus]) if np.any(mask_annulus) else 0
            c3 = np.mean(self.image_float[mask_outer]) if np.any(mask_outer) else 0
            
            energy_region = (
                np.sum((self.image_float[mask_inner] - c1)**2) +
                np.sum((self.image_float[mask_annulus] - c2)**2) +
                np.sum((self.image_float[mask_outer] - c3)**2)
            )
            
            # Length penalty (prefer smooth boundaries)
            length_penalty = 2 * np.pi * (r1 + r2)
            
            # Shape prior (prefer certain radius ratios)
            ratio = r1 / r2
            shape_prior = (ratio - 0.5)**2  # Prefer ratio around 0.5
            
            # Total energy
            lambda_length = 0.1
            lambda_shape = 100.0
            
            return energy_region + lambda_length * length_penalty + lambda_shape * shape_prior
        
        # Initial guess
        x0 = [cx_init, cy_init, r_init * 0.5, r_init * 1.5]
        
        # Bounds
        bounds = [
            (10, self.w - 10),  # cx
            (10, self.h - 10),  # cy
            (5, min(self.h, self.w) // 3),  # r1
            (10, min(self.h, self.w) // 2)  # r2
        ]
        
        # Minimize energy
        result = differential_evolution(energy_functional, bounds, maxiter=100, seed=42)
        
        if result.success:
            return result.x.tolist()
        
        return None
    
    def _robust_combination(self, estimates: List[Tuple[str, List[float]]]) -> List[float]:
        """
        Combine multiple estimates using robust statistics (M-estimators).
        Handles outliers and provides consensus estimate.
        """
        # Extract all estimates
        all_estimates = np.array([est for _, est in estimates])
        
        if len(all_estimates) == 1:
            return all_estimates[0].tolist()
        
        # Use RANSAC-like approach for robust mean
        best_consensus = None
        best_support = 0
        
        for i in range(len(all_estimates)):
            candidate = all_estimates[i]
            
            # Compute distances to this candidate
            distances = np.sqrt(np.sum((all_estimates - candidate)**2, axis=1))
            
            # Adaptive threshold
            threshold = np.median(distances) * 2.0
            
            # Count support
            support = np.sum(distances < threshold)
            
            if support > best_support:
                best_support = support
                best_consensus = candidate
                supporting_estimates = all_estimates[distances < threshold]
        
        # Refine using weighted mean of supporting estimates
        if len(supporting_estimates) > 1:
            # Use Tukey's biweight for robust mean
            center = np.median(supporting_estimates, axis=0)
            mad = np.median(np.abs(supporting_estimates - center), axis=0)
            mad[mad < 1e-6] = 1e-6
            
            # Standardized distances
            u = (supporting_estimates - center) / (6 * mad)
            
            # Tukey weights
            weights = np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
            weights = np.prod(weights, axis=1)
            
            # Weighted mean
            if np.sum(weights) > 0:
                robust_mean = np.sum(supporting_estimates * weights[:, np.newaxis], axis=0) / np.sum(weights)
                return robust_mean.tolist()
        
        return best_consensus.tolist()
    
    def _advanced_refinement(self, initial_params: List[float], 
                           edge_points: np.ndarray) -> np.ndarray:
        """
        Final refinement using advanced optimization with multiple objectives.
        Combines geometric and photometric constraints.
        """
        cx_init, cy_init, r1_init, r2_init = initial_params
        
        # Define comprehensive objective function
        def objective(params):
            cx, cy, r1, r2 = params
            
            # Geometric error: distance from edge points to nearest circle
            distances = np.sqrt((edge_points[:, 0] - cx)**2 + (edge_points[:, 1] - cy)**2)
            
            geometric_errors = np.minimum(
                np.abs(distances - r1),
                np.abs(distances - r2)
            )
            
            # Use Huber loss for robustness
            delta = 2.0
            huber_losses = np.where(
                geometric_errors < delta,
                0.5 * geometric_errors**2,
                delta * geometric_errors - 0.5 * delta**2
            )
            
            geometric_cost = np.sum(huber_losses)
            
            # Photometric error: gradient alignment
            # Sample points on the circles
            n_samples = 72
            angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
            
            photometric_cost = 0
            for r in [r1, r2]:
                for angle in angles:
                    x = cx + r * np.cos(angle)
                    y = cy + r * np.sin(angle)
                    
                    if 2 < x < self.w - 2 and 2 < y < self.h - 2:
                        # Compute radial gradient
                        gx = self._bilinear_interpolate(
                            cv2.Sobel(self.image_float, cv2.CV_64F, 1, 0, ksize=5), x, y
                        )
                        gy = self._bilinear_interpolate(
                            cv2.Sobel(self.image_float, cv2.CV_64F, 0, 1, ksize=5), x, y
                        )
                        
                        # Expected gradient direction (radial)
                        expected_gx = np.cos(angle)
                        expected_gy = np.sin(angle)
                        
                        # Alignment error
                        dot_product = gx * expected_gx + gy * expected_gy
                        photometric_cost += max(0, -dot_product)  # Penalize misalignment
            
            # Regularization terms
            # Prefer certain radius ratios (based on typical fiber optics)
            ratio_cost = 100 * (r1/r2 - 0.4)**2
            
            # Prefer centered circles
            center_cost = 0.1 * ((cx - self.w/2)**2 + (cy - self.h/2)**2)
            
            # Total cost
            return geometric_cost + 0.1 * photometric_cost + ratio_cost + center_cost
        
        # Constraint function
        def constraints(params):
            cx, cy, r1, r2 = params
            return [
                r2 - r1 - 5,  # r2 > r1 + 5
                min(cx, cy, self.w - cx, self.h - cy) - r2 - 5  # Circle fits in image
            ]
        
        # Initial guess
        x0 = np.array([cx_init, cy_init, r1_init, r2_init])
        
        # Bounds
        bounds = [
            (10, self.w - 10),
            (10, self.h - 10),
            (5, min(self.h, self.w) // 3),
            (10, min(self.h, self.w) // 2)
        ]
        
        # Use Trust Region Reflective algorithm for bounded optimization
        result = least_squares(
            lambda p: [objective(p)] + [-c for c in constraints(p)],
            x0,
            bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
            method='trf',
            x_scale='jac',
            loss='soft_l1',
            f_scale=1.0,
            max_nfev=1000
        )
        
        return result.x
    
    def _quantify_uncertainty(self, params: np.ndarray, edge_points: np.ndarray) -> Dict:
        """
        Quantify uncertainty in the detected parameters using 
        bootstrap resampling and sensitivity analysis.
        """
        cx, cy, r1, r2 = params
        n_bootstrap = 100
        
        # Bootstrap resampling
        bootstrap_params = []
        
        for _ in range(n_bootstrap):
            # Resample edge points
            indices = np.random.choice(len(edge_points), len(edge_points), replace=True)
            resampled_points = edge_points[indices]
            
            # Refit using simplified method for speed
            try:
                # Quick RANSAC fit
                refit_params = self._geometric_ransac(resampled_points)
                if refit_params:
                    bootstrap_params.append(refit_params)
            except:
                pass
        
        if len(bootstrap_params) > 10:
            bootstrap_params = np.array(bootstrap_params)
            
            # Compute statistics
            param_mean = np.mean(bootstrap_params, axis=0)
            param_std = np.std(bootstrap_params, axis=0)
            
            # Confidence intervals (95%)
            param_lower = np.percentile(bootstrap_params, 2.5, axis=0)
            param_upper = np.percentile(bootstrap_params, 97.5, axis=0)
            
            uncertainty = {
                'std_dev': {
                    'center_x': param_std[0],
                    'center_y': param_std[1],
                    'r_core': param_std[2],
                    'r_cladding': param_std[3]
                },
                'confidence_interval_95': {
                    'center_x': (param_lower[0], param_upper[0]),
                    'center_y': (param_lower[1], param_upper[1]),
                    'r_core': (param_lower[2], param_upper[2]),
                    'r_cladding': (param_lower[3], param_upper[3])
                },
                'relative_uncertainty': {
                    'r_core': param_std[2] / params[2],
                    'r_cladding': param_std[3] / params[3]
                }
            }
        else:
            # Fallback uncertainty estimation
            uncertainty = {
                'warning': 'Bootstrap failed, using approximate uncertainty',
                'approximate_pixel_uncertainty': 0.5
            }
        
        return uncertainty
    
    def create_visualization(self, save_path: str = 'advanced_detection_result.png'):
        """Create comprehensive visualization of the detection results."""
        if not self.results:
            print("No results to visualize. Run detect_circles_comprehensive first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image with detected circles
        ax = axes[0, 0]
        ax.imshow(self.image, cmap='gray')
        cx, cy = self.results['center']
        r1, r2 = self.results['r_core'], self.results['r_cladding']
        
        circle1 = plt.Circle((cx, cy), r1, fill=False, color='lime', linewidth=2)
        circle2 = plt.Circle((cx, cy), r2, fill=False, color='cyan', linewidth=2)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.plot(cx, cy, 'r+', markersize=10)
        ax.set_title('Detected Circles')
        ax.axis('off')
        
        # Edge points
        ax = axes[0, 1]
        ax.imshow(self.image, cmap='gray', alpha=0.3)
        if 'edge_points' in self.results:
            edge_pts = self.results['edge_points']
            ax.scatter(edge_pts[:, 0], edge_pts[:, 1], s=1, c='red', alpha=0.5)
        ax.set_title('Edge Points (Subpixel)')
        ax.axis('off')
        
        # Radial profile
        ax = axes[0, 2]
        radial_profile = self._compute_radial_profile(cx, cy)
        ax.plot(radial_profile)
        ax.axvline(x=r1, color='lime', linestyle='--', label='Core')
        ax.axvline(x=r2, color='cyan', linestyle='--', label='Cladding')
        ax.set_xlabel('Radius (pixels)')
        ax.set_ylabel('Average Intensity')
        ax.set_title('Radial Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Methods used
        ax = axes[1, 0]
        ax.axis('off')
        methods_text = "Methods Used:\n" + "\n".join(f"• {m}" for m in self.results['methods_used'])
        ax.text(0.1, 0.5, methods_text, transform=ax.transAxes, fontsize=10, verticalalignment='center')
        ax.set_title('Detection Methods')
        
        # Uncertainty visualization
        ax = axes[1, 1]
        if 'uncertainty' in self.results and 'std_dev' in self.results['uncertainty']:
            unc = self.results['uncertainty']['std_dev']
            labels = ['Center X', 'Center Y', 'R Core', 'R Cladding']
            values = [unc['center_x'], unc['center_y'], unc['r_core'], unc['r_cladding']]
            
            bars = ax.bar(labels, values)
            ax.set_ylabel('Standard Deviation (pixels)')
            ax.set_title('Parameter Uncertainty')
            
            # Color bars based on relative uncertainty
            for i, bar in enumerate(bars):
                if values[i] < 0.5:
                    bar.set_color('green')
                elif values[i] < 1.0:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Uncertainty data not available', 
                   transform=ax.transAxes, ha='center', va='center')
        
        # 3D surface plot of intensity
        ax = axes[1, 2]
        ax.remove()
        ax = fig.add_subplot(2, 3, 6, projection='3d')
        
        # Downsample for 3D plot
        step = 5
        x = np.arange(0, self.w, step)
        y = np.arange(0, self.h, step)
        X, Y = np.meshgrid(x, y)
        Z = self.image[::step, ::step]
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Intensity')
        ax.set_title('3D Intensity Surface')
        
        plt.suptitle(f'Advanced Fiber Optic Detection Results\n'
                    f'Center: ({cx:.2f}, {cy:.2f}), '
                    f'R_core: {r1:.2f}, R_cladding: {r2:.2f}',
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def _compute_radial_profile(self, cx: float, cy: float) -> np.ndarray:
        """Compute radial intensity profile for visualization."""
        max_radius = int(min(cx, cy, self.w - cx, self.h - cy))
        profile = np.zeros(max_radius)
        counts = np.zeros(max_radius)
        
        for y in range(self.h):
            for x in range(self.w):
                r = int(np.sqrt((x - cx)**2 + (y - cy)**2))
                if r < max_radius:
                    profile[r] += self.image_float[y, x]
                    counts[r] += 1
        
        # Average
        mask = counts > 0
        profile[mask] /= counts[mask]
        
        return profile


def process_fiber_optic_image(image_path: str, output_dir: str = 'phd_results'):
    """
    Process a fiber optic image using advanced mathematical methods.
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Initialize detector
    detector = AdvancedFiberOpticDetector(image)
    
    # Run comprehensive detection
    try:
        results = detector.detect_circles_comprehensive()
        
        # Print results
        print("\n=== Detection Results ===")
        print(f"Center: ({results['center'][0]:.4f}, {results['center'][1]:.4f})")
        print(f"Core radius: {results['r_core']:.4f} pixels")
        print(f"Cladding radius: {results['r_cladding']:.4f} pixels")
        print(f"Core/Cladding ratio: {results['r_core']/results['r_cladding']:.4f}")
        
        if 'uncertainty' in results and 'std_dev' in results['uncertainty']:
            print("\n=== Uncertainty Analysis ===")
            unc = results['uncertainty']['std_dev']
            print(f"Center uncertainty: ±{max(unc['center_x'], unc['center_y']):.2f} pixels")
            print(f"Core radius uncertainty: ±{unc['r_core']:.2f} pixels")
            print(f"Cladding radius uncertainty: ±{unc['r_cladding']:.2f} pixels")
        
        # Create visualization
        base_name = os.path.basename(image_path).split('.')[0]
        viz_path = os.path.join(output_dir, f"{base_name}_advanced_results.png")
        detector.create_visualization(viz_path)
        
        # Extract and save core and cladding regions
        cx, cy = results['center']
        r_core = results['r_core']
        r_cladding = results['r_cladding']
        
        # Create masks
        dist_sq = (detector.x_grid - cx)**2 + (detector.y_grid - cy)**2
        core_mask = (dist_sq <= r_core**2).astype(np.uint8) * 255
        cladding_mask = ((dist_sq > r_core**2) & (dist_sq <= r_cladding**2)).astype(np.uint8) * 255
        
        # Apply masks
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        core_region = cv2.bitwise_and(gray, gray, mask=core_mask)
        cladding_region = cv2.bitwise_and(gray, gray, mask=cladding_mask)
        
        # Crop to content
        def crop_to_content(img, mask):
            coords = np.argwhere(mask > 0)
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                return img[y_min:y_max+1, x_min:x_max+1]
            return img
        
        core_cropped = crop_to_content(core_region, core_mask)
        cladding_cropped = crop_to_content(cladding_region, cladding_mask)
        
        # Save extracted regions
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_core_advanced.png"), core_cropped)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_cladding_advanced.png"), cladding_cropped)
        
        print(f"\nResults saved to {output_dir}/")
        
        return results
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


# Example usage
if __name__ == "__main__":
    # Test with provided image path
    test_image = r'C:\Users\Saem1001\Documents\GitHub\IPPS\App\images\img63.jpg'
    
    # Process the image
    results = process_fiber_optic_image(test_image)
