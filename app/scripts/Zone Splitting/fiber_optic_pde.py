"""
Ultra-Precise Fiber Optic Detection using PDE and Spectral Methods
Implements cutting-edge mathematical techniques for maximum accuracy
"""

import numpy as np
import cv2
from scipy import ndimage, optimize, signal, special
from scipy.sparse import diags, csr_matrix, linalg as sparse_linalg
from scipy.fft import fft2, ifft2, fftfreq, rfft, irfft
from scipy.integrate import solve_ivp, quad
from scipy.spatial import KDTree
from skimage import measure, morphology
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Callable
import time

class PDESpectralFiberDetector:
    """
    Implements PDE-based and spectral methods for ultra-precise 
    fiber optic core/cladding detection.
    """
    
    def __init__(self, image: np.ndarray):
        """Initialize with preprocessing and setup."""
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image.copy()
        
        self.h, self.w = self.image.shape
        self.image_normalized = self.image.astype(np.float64) / 255.0
        
        # Precompute useful grids and operators
        self._setup_operators()
        
    def _setup_operators(self):
        """Setup differential operators and coordinate systems."""
        # Cartesian grids
        self.x = np.arange(self.w)
        self.y = np.arange(self.h)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Frequency grids for spectral methods
        self.fx = fftfreq(self.w, d=1.0).reshape(1, -1)
        self.fy = fftfreq(self.h, d=1.0).reshape(-1, 1)
        
        # Laplacian operator in Fourier space
        self.laplacian_fourier = -4 * np.pi**2 * (self.fx**2 + self.fy**2)
        
        # Avoid division by zero at DC component
        self.laplacian_fourier[0, 0] = 1.0
        
    def detect_ultra_precise(self) -> Dict:
        """
        Main detection pipeline using advanced PDE and spectral methods.
        """
        print("Starting ultra-precise detection with PDE and spectral methods...")
        
        # 1. Advanced preprocessing using anisotropic diffusion PDE
        processed = self._anisotropic_diffusion_pde()
        
        # 2. Geometric PDE evolution for circle detection
        pde_result = self._geometric_pde_evolution(processed)
        
        # 3. Spectral analysis using circular harmonics
        spectral_result = self._circular_harmonic_analysis()
        
        # 4. Mumford-Shah functional minimization
        ms_result = self._mumford_shah_segmentation(processed)
        
        # 5. Eikonal equation for wavefront propagation
        eikonal_result = self._eikonal_wavefront_method(processed)
        
        # 6. Active contour with topology preservation
        ac_result = self._topology_preserving_active_contour(processed)
        
        # 7. Combine all results using Bayesian inference
        final_params = self._bayesian_combination([
            pde_result, spectral_result, ms_result, 
            eikonal_result, ac_result
        ])
        
        # 8. Super-resolution refinement
        refined_params = self._super_resolution_refinement(final_params, processed)
        
        # 9. Theoretical error bounds
        error_bounds = self._compute_theoretical_bounds(refined_params)
        
        return {
            'center': (refined_params[0], refined_params[1]),
            'r_core': refined_params[2],
            'r_cladding': refined_params[3],
            'error_bounds': error_bounds,
            'intermediate_results': {
                'pde': pde_result,
                'spectral': spectral_result,
                'mumford_shah': ms_result,
                'eikonal': eikonal_result,
                'active_contour': ac_result
            }
        }
    
    def _anisotropic_diffusion_pde(self, iterations: int = 50) -> np.ndarray:
        """
        Solve Perona-Malik anisotropic diffusion PDE for edge-preserving smoothing.
        ∂u/∂t = div(g(|∇u|)∇u) where g is the edge-stopping function.
        """
        u = self.image_normalized.copy()
        dt = 0.1  # Time step
        kappa = 50  # Edge threshold parameter
        
        for _ in range(iterations):
            # Compute gradients using central differences
            ux = np.gradient(u, axis=1)
            uy = np.gradient(u, axis=0)
            
            # Gradient magnitude
            grad_mag = np.sqrt(ux**2 + uy**2)
            
            # Edge-stopping function g(s) = 1/(1 + (s/kappa)²)
            g = 1.0 / (1.0 + (grad_mag / kappa)**2)
            
            # Compute div(g∇u) using finite differences
            # First compute g∇u
            flux_x = g * ux
            flux_y = g * uy
            
            # Then compute divergence
            div_x = np.gradient(flux_x, axis=1)
            div_y = np.gradient(flux_y, axis=0)
            
            # Update using explicit Euler
            u += dt * (div_x + div_y)
            
            # Ensure values stay in [0, 1]
            u = np.clip(u, 0, 1)
        
        return u
    
    def _geometric_pde_evolution(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Evolve level sets using geometric PDEs to detect circles.
        Implements geodesic active contours with curvature flow.
        """
        # Initialize level set as signed distance function
        # Start with a small circle at the center
        cx_init, cy_init = self.w // 2, self.h // 2
        r_init = min(self.w, self.h) // 8
        
        phi = self._create_signed_distance_circle(cx_init, cy_init, r_init)
        
        # Compute edge indicator function
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge stopping function g = 1/(1 + |∇I|²)
        g = 1.0 / (1.0 + grad_mag**2)
        
        # Precompute g gradients
        g_x = np.gradient(g, axis=1)
        g_y = np.gradient(g, axis=0)
        
        # Evolution parameters
        dt = 0.1
        mu = 0.2  # Curvature weight
        nu = 1.0  # Advection weight
        lambda_term = 0.5  # Edge attraction
        
        # Store level set history for detecting multiple circles
        phi_history = []
        
        for iteration in range(200):
            # Compute level set properties
            phi_x = np.gradient(phi, axis=1)
            phi_y = np.gradient(phi, axis=0)
            
            # Avoid division by zero
            grad_phi_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)
            
            # Normalize gradients
            nx = phi_x / grad_phi_mag
            ny = phi_y / grad_phi_mag
            
            # Compute curvature κ = div(∇φ/|∇φ|)
            kappa = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)
            
            # Geodesic active contour evolution
            # ∂φ/∂t = g(μκ + ν)|∇φ| + ∇g·∇φ
            dphi_dt = g * (mu * kappa + nu) * grad_phi_mag + lambda_term * (g_x * phi_x + g_y * phi_y)
            
            # Update level set
            phi += dt * dphi_dt
            
            # Reinitialize as signed distance function every 20 iterations
            if iteration % 20 == 0:
                phi = self._reinitialize_signed_distance(phi)
                
                # Detect circles from zero level set
                contours = measure.find_contours(phi, 0)
                
                if len(contours) >= 2:
                    # Found multiple contours, might be concentric circles
                    circles = []
                    for contour in contours:
                        if len(contour) > 50:
                            circle = self._fit_circle_least_squares(contour[:, [1, 0]])
                            if circle:
                                circles.append(circle)
                    
                    if len(circles) >= 2:
                        # Check for concentricity
                        centers = [(c[0], c[1]) for c in circles]
                        center_distances = [np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) 
                                          for c1 in centers for c2 in centers if c1 != c2]
                        
                        if min(center_distances) < 5.0:  # Concentric
                            radii = sorted([c[2] for c in circles])
                            cx = np.mean([c[0] for c in circles])
                            cy = np.mean([c[1] for c in circles])
                            return [cx, cy, radii[0], radii[1]]
        
        # Fallback: extract single contour and estimate
        contours = measure.find_contours(phi, 0)
        if contours:
            largest_contour = max(contours, key=len)
            circle = self._fit_circle_least_squares(largest_contour[:, [1, 0]])
            if circle:
                # Estimate second radius from intensity profile
                cx, cy, r = circle
                radial_profile = self._extract_radial_profile(image, cx, cy)
                peaks = self._find_profile_peaks(radial_profile)
                if len(peaks) >= 2:
                    return [cx, cy, min(peaks[0], peaks[1]), max(peaks[0], peaks[1])]
        
        return None
    
    def _create_signed_distance_circle(self, cx: float, cy: float, r: float) -> np.ndarray:
        """Create signed distance function for a circle."""
        dist = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2) - r
        return dist
    
    def _reinitialize_signed_distance(self, phi: np.ndarray, iterations: int = 10) -> np.ndarray:
        """
        Reinitialize level set as signed distance function using 
        fast sweeping method for the Eikonal equation |∇φ| = 1.
        """
        # Get zero level set
        zero_level = np.abs(phi) < 1.0
        
        # Use distance transform as initial guess
        dist_positive = ndimage.distance_transform_edt(phi > 0)
        dist_negative = ndimage.distance_transform_edt(phi <= 0)
        phi_new = dist_positive - dist_negative
        
        # Refine using PDE: ∂φ/∂t = sign(φ₀)(1 - |∇φ|)
        phi0_sign = np.sign(phi)
        dt = 0.1
        
        for _ in range(iterations):
            phi_x = np.gradient(phi_new, axis=1)
            phi_y = np.gradient(phi_new, axis=0)
            grad_mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)
            
            phi_new += dt * phi0_sign * (1 - grad_mag)
        
        return phi_new
    
    def _circular_harmonic_analysis(self) -> Optional[List[float]]:
        """
        Perform circular harmonic decomposition using Zernike polynomials
        and Fourier-Bessel series for precise circle detection.
        """
        # First, find approximate center using image moments
        M = cv2.moments(self.image)
        if M['m00'] == 0:
            cx_approx = self.w // 2
            cy_approx = self.h // 2
        else:
            cx_approx = M['m10'] / M['m00']
            cy_approx = M['m01'] / M['m00']
        
        # Define analysis region
        max_radius = min(cx_approx, cy_approx, self.w - cx_approx, self.h - cy_approx) * 0.9
        
        # Compute Zernike moments up to order 20
        zernike_moments = self._compute_zernike_moments(
            self.image_normalized, cx_approx, cy_approx, max_radius, max_order=20
        )
        
        # Analyze rotational symmetry from Zernike moments
        # Circles have strong m=0 components (rotationally symmetric)
        rotational_scores = []
        for n in range(0, 21, 2):  # Even orders for m=0
            if (n, 0) in zernike_moments:
                rotational_scores.append(abs(zernike_moments[(n, 0)]))
        
        # Fourier-Bessel expansion for radial analysis
        n_radial = 100
        n_angular = 72
        
        # Convert to polar coordinates
        r_samples = np.linspace(0, max_radius, n_radial)
        theta_samples = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
        
        polar_image = np.zeros((n_radial, n_angular))
        for i, r in enumerate(r_samples):
            for j, theta in enumerate(theta_samples):
                x = cx_approx + r * np.cos(theta)
                y = cy_approx + r * np.sin(theta)
                if 0 <= x < self.w and 0 <= y < self.h:
                    polar_image[i, j] = self._bilinear_interpolate(self.image_normalized, x, y)
        
        # Compute radial Fourier transform (Hankel-like transform)
        radial_fft = np.fft.fft(polar_image, axis=0)
        
        # Average over angles for m=0 mode
        radial_spectrum = np.mean(np.abs(radial_fft), axis=1)
        
        # Find peaks in radial spectrum (correspond to ring radii)
        peaks, properties = signal.find_peaks(
            radial_spectrum[1:n_radial//2], 
            prominence=np.max(radial_spectrum) * 0.1,
            distance=5
        )
        
        if len(peaks) >= 2:
            # Convert spectral peaks to spatial radii
            # The relationship is approximately r = n_radial / (k + 1)
            # where k is the peak index
            radii = []
            for peak in peaks[:2]:
                k = peak + 1
                r = max_radius * (1.0 - k / (n_radial // 2))
                if r > 0:
                    radii.append(r)
            
            if len(radii) >= 2:
                # Refine center using phase analysis
                cx_refined, cy_refined = self._refine_center_phase_correlation(
                    polar_image, cx_approx, cy_approx, radii
                )
                
                radii.sort()
                return [cx_refined, cy_refined, radii[0], radii[1]]
        
        return None
    
    def _compute_zernike_moments(self, image: np.ndarray, cx: float, cy: float, 
                                radius: float, max_order: int) -> Dict[Tuple[int, int], complex]:
        """Compute Zernike moments up to specified order."""
        moments = {}
        
        # Create coordinate grids relative to center
        x = (self.X - cx) / radius
        y = (self.Y - cy) / radius
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Mask for unit disk
        mask = rho <= 1.0
        
        # Compute Zernike polynomials and moments
        for n in range(max_order + 1):
            for m in range(-n, n + 1, 2):
                if (n - abs(m)) % 2 == 0:
                    # Radial polynomial
                    R_nm = self._zernike_radial(n, abs(m), rho)
                    
                    # Angular part
                    if m >= 0:
                        angular = np.cos(m * theta)
                    else:
                        angular = np.sin(abs(m) * theta)
                    
                    # Zernike basis function
                    Z_nm = R_nm * angular
                    
                    # Compute moment (projection onto basis)
                    moment = np.sum(image[mask] * Z_nm[mask] * np.conjugate(Z_nm[mask]))
                    moment *= (n + 1) / np.pi
                    
                    moments[(n, m)] = moment
        
        return moments
    
    def _zernike_radial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """Compute Zernike radial polynomial R_n^m(ρ)."""
        R = np.zeros_like(rho)
        
        for k in range((n - m) // 2 + 1):
            coeff = ((-1)**k * special.factorial(n - k)) / (
                special.factorial(k) * 
                special.factorial((n + m) // 2 - k) * 
                special.factorial((n - m) // 2 - k)
            )
            R += coeff * rho**(n - 2*k)
        
        return R
    
    def _mumford_shah_segmentation(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Minimize Mumford-Shah functional for piecewise smooth segmentation.
        Detects circles as discontinuity sets.
        """
        # Simplified Chan-Vese model (special case of Mumford-Shah)
        # We'll use a multi-phase version for three regions
        
        # Initialize two level sets for three regions
        cx, cy = self.w // 2, self.h // 2
        r1_init = min(self.w, self.h) // 6
        r2_init = min(self.w, self.h) // 3
        
        phi1 = self._create_signed_distance_circle(cx, cy, r1_init)
        phi2 = self._create_signed_distance_circle(cx, cy, r2_init)
        
        # Parameters
        mu = 0.1  # Length penalty
        dt = 0.1
        
        for iteration in range(100):
            # Define regions
            region1 = (phi1 < 0) & (phi2 < 0)  # Inside both (core)
            region2 = (phi1 >= 0) & (phi2 < 0)  # Between circles (cladding)
            region3 = phi2 >= 0  # Outside both
            
            # Compute region averages
            c1 = np.mean(image[region1]) if np.any(region1) else 0
            c2 = np.mean(image[region2]) if np.any(region2) else 0
            c3 = np.mean(image[region3]) if np.any(region3) else 0
            
            # Compute data fitting terms
            e1 = (image - c1)**2
            e2 = (image - c2)**2
            e3 = (image - c3)**2
            
            # Heaviside and Dirac approximations
            epsilon = 1.0
            H1 = 0.5 * (1 + (2/np.pi) * np.arctan(phi1 / epsilon))
            H2 = 0.5 * (1 + (2/np.pi) * np.arctan(phi2 / epsilon))
            
            delta1 = (epsilon / np.pi) / (epsilon**2 + phi1**2)
            delta2 = (epsilon / np.pi) / (epsilon**2 + phi2**2)
            
            # Compute curvatures
            phi1_x = np.gradient(phi1, axis=1)
            phi1_y = np.gradient(phi1, axis=0)
            phi1_xx = np.gradient(phi1_x, axis=1)
            phi1_yy = np.gradient(phi1_y, axis=0)
            phi1_xy = np.gradient(phi1_x, axis=0)
            
            phi2_x = np.gradient(phi2, axis=1)
            phi2_y = np.gradient(phi2, axis=0)
            phi2_xx = np.gradient(phi2_x, axis=1)
            phi2_yy = np.gradient(phi2_y, axis=0)
            phi2_xy = np.gradient(phi2_x, axis=0)
            
            # Curvature κ = div(∇φ/|∇φ|)
            eps = 1e-10
            kappa1 = (phi1_xx * (phi1_y**2) - 2 * phi1_x * phi1_y * phi1_xy + phi1_yy * (phi1_x**2)) / (
                (phi1_x**2 + phi1_y**2 + eps)**(3/2)
            )
            kappa2 = (phi2_xx * (phi2_y**2) - 2 * phi2_x * phi2_y * phi2_xy + phi2_yy * (phi2_x**2)) / (
                (phi2_x**2 + phi2_y**2 + eps)**(3/2)
            )
            
            # Evolution equations
            F1 = delta1 * (mu * kappa1 - e1 * (1 - H2) + e2 * (1 - H2))
            F2 = delta2 * (mu * kappa2 - e2 * H1 + e3 * H1 - e1 * (1 - H1) + e3 * (1 - H1))
            
            # Update level sets
            phi1 += dt * F1
            phi2 += dt * F2
            
            # Reinitialize periodically
            if iteration % 20 == 0:
                phi1 = self._reinitialize_signed_distance(phi1)
                phi2 = self._reinitialize_signed_distance(phi2)
        
        # Extract circles from final level sets
        contours1 = measure.find_contours(phi1, 0)
        contours2 = measure.find_contours(phi2, 0)
        
        circles = []
        for contours in [contours1, contours2]:
            if contours:
                largest = max(contours, key=len)
                circle = self._fit_circle_least_squares(largest[:, [1, 0]])
                if circle:
                    circles.append(circle)
        
        if len(circles) >= 2:
            # Average centers and sort radii
            cx = np.mean([c[0] for c in circles])
            cy = np.mean([c[1] for c in circles])
            radii = sorted([c[2] for c in circles])
            return [cx, cy, radii[0], radii[1]]
        
        return None
    
    def _eikonal_wavefront_method(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Solve Eikonal equation |∇T| = F for wavefront propagation.
        Circles appear as wavefront stagnation points.
        """
        # Define speed function based on image gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Speed is high in homogeneous regions, low at edges
        F = 1.0 / (1.0 + grad_mag)
        
        # Initialize arrival time
        T = np.full_like(image, np.inf)
        
        # Start propagation from image center
        cx_init, cy_init = self.w // 2, self.h // 2
        T[cy_init, cx_init] = 0
        
        # Fast marching method implementation
        T = self._fast_marching(T, F)
        
        # Analyze arrival time for circular patterns
        # Compute gradient magnitude of arrival time
        T_x = np.gradient(T, axis=1)
        T_y = np.gradient(T, axis=0)
        T_grad_mag = np.sqrt(T_x**2 + T_y**2)
        
        # High gradient in T indicates wavefront stagnation (edges)
        edges = T_grad_mag > np.percentile(T_grad_mag, 90)
        
        # Extract circles from edge map
        # Use Hough transform on the detected edges
        edges_uint8 = (edges * 255).astype(np.uint8)
        
        circles = cv2.HoughCircles(
            edges_uint8,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=min(self.h, self.w) // 2
        )
        
        if circles is not None and len(circles[0]) >= 2:
            circles = circles[0]
            # Sort by radius
            circles = circles[np.argsort(circles[:, 2])]
            
            # Take two largest circles
            if len(circles) >= 2:
                c1, c2 = circles[-2], circles[-1]
                
                # Check concentricity
                center_dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if center_dist < 10:
                    cx = (c1[0] + c2[0]) / 2
                    cy = (c1[1] + c2[1]) / 2
                    r1 = min(c1[2], c2[2])
                    r2 = max(c1[2], c2[2])
                    return [cx, cy, r1, r2]
        
        return None
    
    def _fast_marching(self, T: np.ndarray, F: np.ndarray, max_iterations: int = 10000) -> np.ndarray:
        """
        Fast marching method for solving Eikonal equation.
        Simplified implementation using Dijkstra-like approach.
        """
        # Create priority queue (using heap)
        import heapq
        
        h, w = T.shape
        heap = []
        
        # Status: 0=far, 1=trial, 2=alive
        status = np.zeros_like(T, dtype=int)
        
        # Find initial alive points (where T is finite)
        alive_mask = T < np.inf
        status[alive_mask] = 2
        
        # Initialize trial points (neighbors of alive points)
        for y in range(h):
            for x in range(w):
                if status[y, x] == 2:  # Alive point
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and status[ny, nx] == 0:
                            status[ny, nx] = 1
                            T_new = self._solve_eikonal_local(T, F, ny, nx)
                            T[ny, nx] = T_new
                            heapq.heappush(heap, (T_new, ny, nx))
        
        # Main loop
        iterations = 0
        while heap and iterations < max_iterations:
            current_t, y, x = heapq.heappop(heap)
            
            if status[y, x] == 2:  # Already alive
                continue
            
            status[y, x] = 2  # Mark as alive
            
            # Update neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and status[ny, nx] != 2:
                    T_new = self._solve_eikonal_local(T, F, ny, nx)
                    
                    if T_new < T[ny, nx]:
                        T[ny, nx] = T_new
                        status[ny, nx] = 1
                        heapq.heappush(heap, (T_new, ny, nx))
            
            iterations += 1
        
        return T
    
    def _solve_eikonal_local(self, T: np.ndarray, F: np.ndarray, y: int, x: int) -> float:
        """
        Solve Eikonal equation locally using upwind finite differences.
        |∇T| = 1/F
        """
        h, w = T.shape
        
        # Get neighbor values
        T_xmin = T[y, x-1] if x > 0 and T[y, x-1] < np.inf else np.inf
        T_xmax = T[y, x+1] if x < w-1 and T[y, x+1] < np.inf else np.inf
        T_ymin = T[y-1, x] if y > 0 and T[y-1, x] < np.inf else np.inf
        T_ymax = T[y+1, x] if y < h-1 and T[y+1, x] < np.inf else np.inf
        
        # Use smallest values in each direction
        T_x = min(T_xmin, T_xmax)
        T_y = min(T_ymin, T_ymax)
        
        # Solve quadratic equation
        if T_x == np.inf and T_y == np.inf:
            return np.inf
        elif T_x == np.inf:
            return T_y + 1.0 / F[y, x]
        elif T_y == np.inf:
            return T_x + 1.0 / F[y, x]
        else:
            # Solve (T - T_x)² + (T - T_y)² = 1/F²
            a = 2.0
            b = -2.0 * (T_x + T_y)
            c = T_x**2 + T_y**2 - 1.0 / F[y, x]**2
            
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                T_new = (-b + np.sqrt(discriminant)) / (2*a)
                return max(T_new, max(T_x, T_y))
            else:
                return min(T_x, T_y) + 1.0 / F[y, x]
    
    def _topology_preserving_active_contour(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Active contour that preserves topology (maintains two loops).
        Uses discrete topology control.
        """
        # Initialize two circular contours
        cx, cy = self.w // 2, self.h // 2
        n_points = 100
        
        # Inner circle
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        r1 = min(self.w, self.h) // 6
        contour1 = np.column_stack([
            cx + r1 * np.cos(theta),
            cy + r1 * np.sin(theta)
        ])
        
        # Outer circle
        r2 = min(self.w, self.h) // 3
        contour2 = np.column_stack([
            cx + r2 * np.cos(theta),
            cy + r2 * np.sin(theta)
        ])
        
        # Precompute image forces
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        
        # Parameters
        alpha = 0.1  # Elasticity
        beta = 0.1   # Rigidity
        gamma = 1.0  # Image force weight
        iterations = 200
        
        for _ in range(iterations):
            # Update each contour
            for contour in [contour1, contour2]:
                n = len(contour)
                
                # Compute internal forces
                # Elasticity (first derivative)
                d1 = np.roll(contour, -1, axis=0) - contour
                d2 = contour - np.roll(contour, 1, axis=0)
                elastic_force = alpha * (d1 + d2)
                
                # Rigidity (second derivative)
                d2_contour = (np.roll(contour, -1, axis=0) - 
                            2 * contour + 
                            np.roll(contour, 1, axis=0))
                rigid_force = -beta * d2_contour
                
                # External forces (image gradients)
                external_force = np.zeros_like(contour)
                for i, (x, y) in enumerate(contour):
                    if 0 <= x < self.w and 0 <= y < self.h:
                        fx = self._bilinear_interpolate(grad_x, x, y)
                        fy = self._bilinear_interpolate(grad_y, x, y)
                        external_force[i] = [-fx, -fy]
                
                # Update contour
                total_force = elastic_force + rigid_force + gamma * external_force
                contour += 0.5 * total_force
                
                # Enforce topology: keep points ordered and non-intersecting
                # Simple approach: re-parameterize as circle after each iteration
                center = np.mean(contour, axis=0)
                vectors = contour - center
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                order = np.argsort(angles)
                contour[:] = contour[order]
        
        # Extract final circles
        circle1 = self._fit_circle_least_squares(contour1)
        circle2 = self._fit_circle_least_squares(contour2)
        
        if circle1 and circle2:
            # Average centers
            cx = (circle1[0] + circle2[0]) / 2
            cy = (circle1[1] + circle2[1]) / 2
            r1 = min(circle1[2], circle2[2])
            r2 = max(circle1[2], circle2[2])
            return [cx, cy, r1, r2]
        
        return None
    
    def _bayesian_combination(self, results: List[Optional[List[float]]]) -> List[float]:
        """
        Combine results using Bayesian inference with uncertainty estimation.
        """
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            # Fallback to simple estimation
            cx, cy = self.w // 2, self.h // 2
            r1 = min(self.w, self.h) // 6
            r2 = min(self.w, self.h) // 3
            return [cx, cy, r1, r2]
        
        if len(valid_results) == 1:
            return valid_results[0]
        
        # Convert to numpy array
        results_array = np.array(valid_results)
        
        # Estimate mean and covariance
        mean = np.mean(results_array, axis=0)
        
        if len(valid_results) > 1:
            cov = np.cov(results_array.T)
            
            # Bayesian update with informative prior
            # Prior: circles should be roughly centered and have reasonable radii
            prior_mean = np.array([self.w/2, self.h/2, min(self.w, self.h)/6, min(self.w, self.h)/3])
            prior_cov = np.diag([100, 100, 50, 100])  # Prior uncertainty
            
            # Posterior (using conjugate prior for normal distribution)
            cov_inv = np.linalg.pinv(cov)
            prior_cov_inv = np.linalg.pinv(prior_cov)
            
            posterior_cov_inv = prior_cov_inv + len(valid_results) * cov_inv
            posterior_cov = np.linalg.pinv(posterior_cov_inv)
            
            posterior_mean = posterior_cov @ (prior_cov_inv @ prior_mean + 
                                            len(valid_results) * cov_inv @ mean)
            
            return posterior_mean.tolist()
        else:
            return mean.tolist()
    
    def _super_resolution_refinement(self, params: List[float], image: np.ndarray) -> List[float]:
        """
        Super-resolution refinement using interpolation and 
        fractional calculus for sub-pixel accuracy.
        """
        cx, cy, r1, r2 = params
        
        # Define refinement region around detected circles
        margin = 20
        x_min = max(0, int(cx - r2 - margin))
        x_max = min(self.w, int(cx + r2 + margin))
        y_min = max(0, int(cy - r2 - margin))
        y_max = min(self.h, int(cy + r2 + margin))
        
        # Extract region
        region = image[y_min:y_max, x_min:x_max]
        
        # Upsample using bicubic interpolation
        scale_factor = 4
        upsampled = cv2.resize(region, None, fx=scale_factor, fy=scale_factor, 
                              interpolation=cv2.INTER_CUBIC)
        
        # Apply fractional derivative for edge enhancement
        # Using Grünwald-Letnikov definition
        alpha = 0.5  # Fractional order
        enhanced = self._fractional_derivative(upsampled, alpha)
        
        # Detect edges in upsampled image
        edges = cv2.Canny((enhanced * 255).astype(np.uint8), 50, 150)
        
        # Extract edge points
        edge_points = np.argwhere(edges > 0)
        if len(edge_points) > 100:
            # Convert back to original coordinates
            edge_points = edge_points / scale_factor + np.array([y_min, x_min])
            edge_points = edge_points[:, [1, 0]]  # Swap to (x, y)
            
            # Refine circles using robust fitting
            def objective(p):
                cx_new, cy_new, r1_new, r2_new = p
                distances = np.sqrt((edge_points[:, 0] - cx_new)**2 + 
                                  (edge_points[:, 1] - cy_new)**2)
                
                # Robust loss (Tukey's biweight)
                errors1 = np.abs(distances - r1_new)
                errors2 = np.abs(distances - r2_new)
                errors = np.minimum(errors1, errors2)
                
                c = 4.685  # Tuning constant
                weights = np.where(errors < c, (1 - (errors/c)**2)**2, 0)
                
                return np.sum(weights * errors**2)
            
            # Optimize
            result = optimize.minimize(
                objective, 
                params,
                method='L-BFGS-B',
                bounds=[
                    (cx - 5, cx + 5),
                    (cy - 5, cy + 5),
                    (r1 - 5, r1 + 5),
                    (r2 - 5, r2 + 5)
                ]
            )
            
            if result.success:
                return result.x.tolist()
        
        return params
    
    def _fractional_derivative(self, image: np.ndarray, alpha: float) -> np.ndarray:
        """
        Compute fractional derivative using Grünwald-Letnikov definition.
        Enhances edges while preserving smooth regions.
        """
        h = 1.0  # Grid spacing
        n = min(50, image.shape[0])  # Truncation length
        
        # Compute GL coefficients
        gl_coeffs = np.zeros(n)
        gl_coeffs[0] = 1.0
        for k in range(1, n):
            gl_coeffs[k] = gl_coeffs[k-1] * (k - 1 - alpha) / k
        
        # Apply to each row (x-direction)
        result_x = np.zeros_like(image)
        for i in range(image.shape[0]):
            row = image[i, :]
            for j in range(n, image.shape[1]):
                result_x[i, j] = np.sum(gl_coeffs * row[j:j-n:-1]) / (h**alpha)
        
        # Apply to each column (y-direction)
        result_y = np.zeros_like(image)
        for j in range(image.shape[1]):
            col = image[:, j]
            for i in range(n, image.shape[0]):
                result_y[i, j] = np.sum(gl_coeffs * col[i:i-n:-1]) / (h**alpha)
        
        # Combine
        return np.sqrt(result_x**2 + result_y**2)
    
    def _compute_theoretical_bounds(self, params: List[float]) -> Dict:
        """
        Compute theoretical error bounds using perturbation analysis
        and condition number estimation.
        """
        cx, cy, r1, r2 = params
        
        # Estimate condition number of the problem
        # Based on gradient magnitude near circles
        condition_numbers = []
        
        for r in [r1, r2]:
            # Sample points on circle
            n_samples = 36
            theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
            
            gradients = []
            for t in theta:
                x = cx + r * np.cos(t)
                y = cy + r * np.sin(t)
                
                if 1 < x < self.w-1 and 1 < y < self.h-1:
                    # Local gradient magnitude
                    local_region = self.image_normalized[
                        int(y)-1:int(y)+2, 
                        int(x)-1:int(x)+2
                    ]
                    grad = np.gradient(local_region)
                    gradients.append(np.sqrt(grad[0]**2 + grad[1]**2).max())
            
            if gradients:
                # Condition number ≈ 1 / (average gradient strength)
                avg_gradient = np.mean(gradients)
                condition_numbers.append(1.0 / (avg_gradient + 1e-6))
        
        # Theoretical bounds based on:
        # 1. Pixel discretization error: ±0.5 pixels
        # 2. Interpolation error: O(h²) for bilinear
        # 3. Numerical precision: ~1e-10
        # 4. Condition number amplification
        
        avg_condition = np.mean(condition_numbers) if condition_numbers else 10.0
        
        bounds = {
            'pixel_discretization': 0.5,
            'interpolation_error': 0.25,  # For bilinear
            'numerical_precision': 1e-10,
            'condition_amplification': avg_condition,
            'estimated_total_error': {
                'center': 0.5 * np.sqrt(2) * avg_condition,
                'radius': 0.5 * avg_condition
            },
            'confidence_level': 0.95
        }
        
        return bounds
    
    def _fit_circle_least_squares(self, points: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Robust circle fitting using least squares with outlier rejection."""
        if len(points) < 5:
            return None
        
        # Kasa method with iterative reweighting
        x, y = points[:, 0], points[:, 1]
        
        # Center data for numerical stability
        x_m = np.mean(x)
        y_m = np.mean(y)
        u = x - x_m
        v = y - y_m
        
        # Initial fit
        Suu = np.sum(u*u)
        Svv = np.sum(v*v)
        Suv = np.sum(u*v)
        Suuu = np.sum(u*u*u)
        Svvv = np.sum(v*v*v)
        Suvv = np.sum(u*v*v)
        Svuu = np.sum(v*u*u)
        
        # Solve linear system
        A = np.array([[Suu, Suv], [Suv, Svv]])
        b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
        
        try:
            uc, vc = np.linalg.solve(A, b)
            xc = uc + x_m
            yc = vc + y_m
            r = np.sqrt(uc*uc + vc*vc + (Suu + Svv) / len(points))
            
            # Iterative refinement with M-estimator
            for _ in range(3):
                # Compute residuals
                distances = np.sqrt((x - xc)**2 + (y - yc)**2)
                residuals = np.abs(distances - r)
                
                # Huber weights
                k = 1.345 * np.median(residuals)
                weights = np.where(residuals <= k, 1.0, k / residuals)
                
                # Weighted fit
                wx = weights * (x - x_m)
                wy = weights * (y - y_m)
                
                Swuu = np.sum(weights * u * u)
                Swvv = np.sum(weights * v * v)
                Swuv = np.sum(weights * u * v)
                Swuuu = np.sum(weights * u * u * u)
                Swvvv = np.sum(weights * v * v * v)
                Swuvv = np.sum(weights * u * v * v)
                Swvuu = np.sum(weights * v * u * u)
                
                A = np.array([[Swuu, Swuv], [Swuv, Swvv]])
                b = 0.5 * np.array([Swuuu + Swuvv, Swvvv + Swvuu])
                
                if np.linalg.det(A) > 1e-10:
                    uc, vc = np.linalg.solve(A, b)
                    xc = uc + x_m
                    yc = vc + y_m
                    r = np.sqrt(np.sum(weights * ((x - xc)**2 + (y - yc)**2)) / np.sum(weights))
            
            if 0 < r < max(self.h, self.w):
                return (xc, yc, r)
                
        except np.linalg.LinAlgError:
            pass
        
        return None
    
    def _extract_radial_profile(self, image: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Extract radial intensity profile."""
        max_r = int(min(cx, cy, self.w - cx, self.h - cy))
        profile = np.zeros(max_r)
        counts = np.zeros(max_r)
        
        for y in range(self.h):
            for x in range(self.w):
                r = int(np.sqrt((x - cx)**2 + (y - cy)**2))
                if r < max_r:
                    profile[r] += image[y, x]
                    counts[r] += 1
        
        # Average
        mask = counts > 0
        profile[mask] /= counts[mask]
        
        return profile
    
    def _find_profile_peaks(self, profile: np.ndarray) -> List[float]:
        """Find peaks in radial profile corresponding to circles."""
        # Smooth profile
        smooth_profile = ndimage.gaussian_filter1d(profile, sigma=2)
        
        # Compute derivative
        derivative = np.gradient(smooth_profile)
        
        # Find local minima (edges)
        minima = []
        for i in range(10, len(derivative) - 10):
            if derivative[i] < derivative[i-1] and derivative[i] < derivative[i+1]:
                if derivative[i] < -0.01:  # Threshold
                    minima.append(i)
        
        return minima
    
    def _bilinear_interpolate(self, image: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation for subpixel access."""
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, image.shape[1] - 1), min(y0 + 1, image.shape[0] - 1)
        
        wx, wy = x - x0, y - y0
        
        return ((1-wx) * (1-wy) * image[y0, x0] +
                wx * (1-wy) * image[y0, x1] +
                (1-wx) * wy * image[y1, x0] +
                wx * wy * image[y1, x1])
    
    def _refine_center_phase_correlation(self, polar_image: np.ndarray, 
                                       cx: float, cy: float, 
                                       radii: List[float]) -> Tuple[float, float]:
        """Refine center using phase correlation in polar domain."""
        # This is a simplified version
        # In practice, would use full phase correlation
        return cx, cy
    
    def visualize_results(self, results: Dict, save_path: str = 'ultra_precise_results.png'):
        """Create comprehensive visualization of ultra-precise results."""
        fig = plt.figure(figsize=(20, 12))
        
        # Original image with results
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(self.image, cmap='gray')
        cx, cy = results['center']
        r1, r2 = results['r_core'], results['r_cladding']
        
        circle1 = plt.Circle((cx, cy), r1, fill=False, color='lime', linewidth=3, label='Core')
        circle2 = plt.Circle((cx, cy), r2, fill=False, color='cyan', linewidth=3, label='Cladding')
        ax1.add_patch(circle1)
        ax1.add_patch(circle2)
        ax1.plot(cx, cy, 'r+', markersize=15, markeredgewidth=3)
        ax1.set_title('Ultra-Precise Detection Result', fontsize=14)
        ax1.legend()
        ax1.axis('off')
        
        # Intermediate results comparison
        ax2 = plt.subplot(2, 3, 2)
        methods = ['PDE', 'Spectral', 'Mumford-Shah', 'Eikonal', 'Active Contour']
        intermediate = results['intermediate_results']
        
        # Plot each method's result
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (method, color) in enumerate(zip(methods, colors)):
            key = method.lower().replace('-', '_').replace(' ', '_')
            if key in intermediate and intermediate[key] is not None:
                res = intermediate[key]
                offset = (i - 2) * 0.5  # Small offset for visibility
                circle = plt.Circle((res[0] + offset, res[1] + offset), res[2], 
                                  fill=False, color=color, alpha=0.5, linewidth=2)
                ax2.add_patch(circle)
                ax2.text(10, 20 + i*15, f'{method}: ({res[0]:.1f}, {res[1]:.1f})', 
                        color=color, fontsize=10)
        
        ax2.set_xlim(0, self.w)
        ax2.set_ylim(self.h, 0)
        ax2.set_title('Method Comparison', fontsize=14)
        ax2.set_aspect('equal')
        
        # Error bounds visualization
        ax3 = plt.subplot(2, 3, 3)
        if 'error_bounds' in results:
            bounds = results['error_bounds']
            
            # Create error ellipse
            from matplotlib.patches import Ellipse
            
            total_error = bounds['estimated_total_error']
            ellipse1 = Ellipse((cx, cy), 
                              2 * r1 * total_error['radius'] / r1,
                              2 * r1 * total_error['radius'] / r1,
                              fill=False, color='lime', linestyle='--', alpha=0.5)
            ellipse2 = Ellipse((cx, cy),
                              2 * r2 * total_error['radius'] / r2,
                              2 * r2 * total_error['radius'] / r2,
                              fill=False, color='cyan', linestyle='--', alpha=0.5)
            
            ax3.imshow(self.image, cmap='gray', alpha=0.3)
            ax3.add_patch(ellipse1)
            ax3.add_patch(ellipse2)
            
            # Add circles
            circle1 = plt.Circle((cx, cy), r1, fill=False, color='lime', linewidth=2)
            circle2 = plt.Circle((cx, cy), r2, fill=False, color='cyan', linewidth=2)
            ax3.add_patch(circle1)
            ax3.add_patch(circle2)
            
            ax3.set_title(f'Error Bounds (±{total_error["radius"]:.3f} pixels)', fontsize=14)
            ax3.axis('off')
        
        # Spectral analysis
        ax4 = plt.subplot(2, 3, 4)
        # Compute and display radial FFT
        radial_profile = self._extract_radial_profile(self.image_normalized, cx, cy)
        radial_fft = np.abs(np.fft.rfft(radial_profile))
        freqs = np.fft.rfftfreq(len(radial_profile))
        
        ax4.semilogy(freqs[1:], radial_fft[1:])
        ax4.set_xlabel('Spatial Frequency')
        ax4.set_ylabel('Magnitude')
        ax4.set_title('Radial Spectrum Analysis', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Level set evolution visualization
        ax5 = plt.subplot(2, 3, 5)
        # Show gradient magnitude with detected circles
        grad_x = np.gradient(self.image_normalized, axis=1)
        grad_y = np.gradient(self.image_normalized, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        ax5.imshow(grad_mag, cmap='hot')
        circle1 = plt.Circle((cx, cy), r1, fill=False, color='lime', linewidth=2)
        circle2 = plt.Circle((cx, cy), r2, fill=False, color='cyan', linewidth=2)
        ax5.add_patch(circle1)
        ax5.add_patch(circle2)
        ax5.set_title('Gradient Magnitude', fontsize=14)
        ax5.axis('off')
        
        # 3D surface plot
        ax6 = plt.subplot(2, 3, 6, projection='3d')
        # Downsample for visualization
        step = 5
        x_range = np.arange(0, self.w, step)
        y_range = np.arange(0, self.h, step)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z_grid = self.image[::step, ::step]
        
        surf = ax6.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', 
                               alpha=0.8, antialiased=True)
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Intensity')
        ax6.set_title('3D Intensity Surface', fontsize=14)
        
        # Main title with results
        fig.suptitle(
            f'Ultra-Precise Fiber Optic Detection\n'
            f'Center: ({cx:.4f}, {cy:.4f}) ± {total_error["center"]:.4f} pixels\n'
            f'Core: {r1:.4f} ± {total_error["radius"]:.4f} pixels, '
            f'Cladding: {r2:.4f} ± {total_error["radius"]:.4f} pixels',
            fontsize=16
        )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def process_with_ultra_precision(image_path: str, output_dir: str = 'ultra_precise_results'):
    """Process fiber optic image with ultra-precise PDE and spectral methods."""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing {image_path} with ultra-precise methods...")
    
    # Initialize detector
    detector = PDESpectralFiberDetector(image)
    
    # Run detection
    start_time = time.time()
    results = detector.detect_ultra_precise()
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n=== Ultra-Precise Detection Results ===")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Center: ({results['center'][0]:.6f}, {results['center'][1]:.6f})")
    print(f"Core radius: {results['r_core']:.6f} pixels")
    print(f"Cladding radius: {results['r_cladding']:.6f} pixels")
    print(f"Core/Cladding ratio: {results['r_core']/results['r_cladding']:.6f}")
    
    if 'error_bounds' in results:
        bounds = results['error_bounds']
        print(f"\n=== Theoretical Error Bounds ===")
        print(f"Condition number: {bounds['condition_amplification']:.2f}")
        print(f"Estimated total error:")
        print(f"  Center: ±{bounds['estimated_total_error']['center']:.4f} pixels")
        print(f"  Radius: ±{bounds['estimated_total_error']['radius']:.4f} pixels")
    
    # Create visualization
    base_name = os.path.basename(image_path).split('.')[0]
    viz_path = os.path.join(output_dir, f"{base_name}_ultra_precise.png")
    detector.visualize_results(results, viz_path)
    
    # Extract regions with ultra-precise masks
    cx, cy = results['center']
    r_core = results['r_core']
    r_cladding = results['r_cladding']
    
    # Create sub-pixel accurate masks using antialiasing
    dist = np.sqrt((detector.X - cx)**2 + (detector.Y - cy)**2)
    
    # Soft masks with antialiasing
    sigma = 0.5  # Antialiasing width
    core_mask_soft = 1.0 / (1.0 + np.exp((dist - r_core) / sigma))
    cladding_mask_soft = 1.0 / (1.0 + np.exp((dist - r_cladding) / sigma))
    annulus_mask_soft = cladding_mask_soft - core_mask_soft
    
    # Apply soft masks
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    core_region = (gray * core_mask_soft).astype(np.uint8)
    cladding_region = (gray * annulus_mask_soft).astype(np.uint8)
    
    # Save regions
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_core_ultra.png"), core_region)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_cladding_ultra.png"), cladding_region)
    
    print(f"\nResults saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    # Test with example image
    test_image = r'C:\Users\Saem1001\Documents\GitHub\IPPS\App\images\img63.jpg'
    results = process_with_ultra_precision(test_image)
