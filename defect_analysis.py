import numpy as np
import cv2
from scipy import ndimage, signal, stats
from scipy.ndimage import gaussian_filter, median_filter
from scipy.sparse.linalg import svds
from scipy.optimize import minimize
from skimage import morphology, measure, feature
from skimage.restoration import denoise_tv_chambolle
import warnings
warnings.filterwarnings('ignore')

class FiberOpticDefectDetector:
    """
    Advanced detector for scratches and digs in fiber optic end faces.
    Implements multiple PhD-level mathematical methods for robust detection.
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        self.intermediate_results = {}
        
    def detect_defects(self, image):
        """
        Main detection.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Store original
        self.original = image.copy()
        self.height, self.width = image.shape
        
        # Step 1: Advanced preprocessing
        preprocessed = self._advanced_preprocessing(image)
        
        # Step 2: Multi-method scratch detection
        scratch_mask = self._detect_scratches_multimethod(preprocessed)
        
        # Step 3: Multi-method dig detection
        dig_mask = self._detect_digs_multimethod(preprocessed)
        
        # Step 4: Refinement using variational methods
        refined_scratches = self._refine_with_variational_methods(scratch_mask, preprocessed)
        refined_digs = self._refine_with_variational_methods(dig_mask, preprocessed)
        
        # Step 5: False positive reduction
        final_scratches = self._reduce_false_positives_scratches(refined_scratches, preprocessed)
        final_digs = self._reduce_false_positives_digs(refined_digs, preprocessed)
        
        return {
            'scratches': final_scratches,
            'digs': final_digs,
            'combined': np.logical_or(final_scratches, final_digs),
            'intermediate': self.intermediate_results if self.debug else None
        }
    
    def _advanced_preprocessing(self, image):
        """
        Preprocessing
        """
        # 1. Anisotropic diffusion (Perona-Malik)
        diffused = self._anisotropic_diffusion(image.astype(np.float64))
        
        # 2. Total Variation denoising
        tv_denoised = denoise_tv_chambolle(diffused, weight=0.1)
        
        # 3. Coherence-enhancing diffusion
        coherence_enhanced = self._coherence_enhancing_diffusion(tv_denoised)
        
        # 4. Adaptive histogram equalization for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply((coherence_enhanced * 255).astype(np.uint8))
        
        if self.debug:
            self.intermediate_results['preprocessed'] = enhanced
            
        return enhanced.astype(np.float64) / 255.0
    
    def _anisotropic_diffusion(self, image, iterations=10, kappa=50, gamma=0.1):
        """
        Perona-Malik anisotropic diffusion.
        """
        img = image.copy()
        
        for _ in range(iterations):
            # Calculate gradients
            nablaE = np.roll(img, -1, axis=1) - img  # East
            nablaW = np.roll(img, 1, axis=1) - img   # West
            nablaN = np.roll(img, -1, axis=0) - img  # North
            nablaS = np.roll(img, 1, axis=0) - img   # South
            
            # Diffusion coefficient g(|∇I|) = 1 / (1 + (|∇I|/K)²)
            cE = 1.0 / (1.0 + (nablaE/kappa)**2)
            cW = 1.0 / (1.0 + (nablaW/kappa)**2)
            cN = 1.0 / (1.0 + (nablaN/kappa)**2)
            cS = 1.0 / (1.0 + (nablaS/kappa)**2)
            
            # Update
            img += gamma * (cE*nablaE + cW*nablaW + cN*nablaN + cS*nablaS)
            
        return img
    
    def _coherence_enhancing_diffusion(self, image, iterations=5):
        """
        Coherence-enhancing diffusion for linear structures.
        """
        img = image.copy()
        
        for _ in range(iterations):
            # Structure tensor
            J = self._compute_structure_tensor(img)
            
            # Eigenvalues and eigenvectors
            eigenvals, eigenvecs = self._eigen_decomposition_2x2(J)
            
            # Diffusion tensor
            D = self._compute_diffusion_tensor(eigenvals, eigenvecs)
            
            # Apply diffusion
            img = self._apply_tensor_diffusion(img, D)
            
        return img
    
    def _compute_structure_tensor(self, image, sigma=1.0):
        """
        Compute structure tensor J = ∇I ⊗ ∇I.
        """
        # Gradients
        Ix = ndimage.sobel(image, axis=1)
        Iy = ndimage.sobel(image, axis=0)
        
        # Structure tensor components
        Jxx = gaussian_filter(Ix * Ix, sigma)
        Jxy = gaussian_filter(Ix * Iy, sigma)
        Jyy = gaussian_filter(Iy * Iy, sigma)
        
        return np.stack([Jxx, Jxy, Jxy, Jyy], axis=-1).reshape(*image.shape, 2, 2)
    
    def _eigen_decomposition_2x2(self, J):
        """
        Efficient eigendecomposition for 2x2 matrices.
        """
        # Extract components
        a = J[..., 0, 0]
        b = J[..., 0, 1]
        c = J[..., 1, 1]
        
        # Trace and determinant
        trace = a + c
        det = a * c - b * b
        
        # Eigenvalues
        discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
        lambda1 = 0.5 * (trace + discriminant)
        lambda2 = 0.5 * (trace - discriminant)
        
        # Eigenvectors
        v1x = lambda1 - c
        v1y = b
        norm1 = np.sqrt(v1x**2 + v1y**2 + 1e-10)
        v1x /= norm1
        v1y /= norm1
        
        v2x = -v1y
        v2y = v1x
        
        eigenvals = np.stack([lambda1, lambda2], axis=-1)
        eigenvecs = np.stack([v1x, v1y, v2x, v2y], axis=-1).reshape(*J.shape[:-2], 2, 2)
        
        return eigenvals, eigenvecs
    
    def _compute_diffusion_tensor(self, eigenvals, eigenvecs, alpha=0.001):
        """
        Compute diffusion tensor for coherence enhancement.
        """
        lambda1 = eigenvals[..., 0]
        lambda2 = eigenvals[..., 1]
        
        # Coherence measure
        coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10))**2
        
        # Diffusion eigenvalues
        c1 = alpha
        c2 = alpha + (1 - alpha) * np.exp(-1 / (coherence + 1e-10))
        
        # Reconstruct diffusion tensor
        D = np.zeros_like(eigenvecs)
        D[..., 0, 0] = c1 * eigenvecs[..., 0, 0]**2 + c2 * eigenvecs[..., 1, 0]**2
        D[..., 0, 1] = c1 * eigenvecs[..., 0, 0] * eigenvecs[..., 0, 1] + \
                        c2 * eigenvecs[..., 1, 0] * eigenvecs[..., 1, 1]
        D[..., 1, 0] = D[..., 0, 1]
        D[..., 1, 1] = c1 * eigenvecs[..., 0, 1]**2 + c2 * eigenvecs[..., 1, 1]**2
        
        return D
    
    def _apply_tensor_diffusion(self, image, D, dt=0.1):
        """
        Apply tensor-based diffusion.
        """
        # Compute second derivatives
        Ixx = ndimage.sobel(ndimage.sobel(image, axis=1), axis=1)
        Iyy = ndimage.sobel(ndimage.sobel(image, axis=0), axis=0)
        Ixy = ndimage.sobel(ndimage.sobel(image, axis=1), axis=0)
        
        # Diffusion update
        div = D[..., 0, 0] * Ixx + 2 * D[..., 0, 1] * Ixy + D[..., 1, 1] * Iyy
        
        return image + dt * div
    
    def _detect_scratches_multimethod(self, image):
        """
        Detect scratches using multiple methods.
        """
        methods = []
        
        # 1. Hessian ridge detection
        hessian_ridges = self._hessian_ridge_detection(image)
        methods.append(hessian_ridges)
        
        # 2. Frangi vesselness filter
        frangi = self._frangi_vesselness(image)
        methods.append(frangi)
        
        # 3. Phase congruency
        phase_congruency = self._phase_congruency(image)
        methods.append(phase_congruency)
        
        # 4. Radon transform peak detection
        radon_lines = self._radon_line_detection(image)
        methods.append(radon_lines)
        
        # 5. Directional filter bank
        directional = self._directional_filter_bank(image)
        methods.append(directional)
        
        # Combine using weighted voting
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]
        combined = np.zeros_like(image)
        for method, weight in zip(methods, weights):
            combined += weight * method
            
        # Adaptive thresholding
        threshold = np.mean(combined) + 2 * np.std(combined)
        scratch_mask = combined > threshold
        
        # Morphological refinement
        scratch_mask = self._morphological_scratch_refinement(scratch_mask)
        
        if self.debug:
            self.intermediate_results['scratch_methods'] = methods
            self.intermediate_results['scratch_combined'] = combined
            
        return scratch_mask
    
    def _hessian_ridge_detection(self, image, scales=[1, 2, 3]):
        """
        Multi-scale Hessian ridge detection.
        """
        ridge_response = np.zeros_like(image)
        
        for scale in scales:
            # Gaussian smoothing at scale
            smoothed = gaussian_filter(image, scale)
            
            # Hessian matrix
            Hxx = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=1)
            Hyy = ndimage.sobel(ndimage.sobel(smoothed, axis=0), axis=0)
            Hxy = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=0)
            
            # Eigenvalues
            trace = Hxx + Hyy
            det = Hxx * Hyy - Hxy * Hxy
            discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
            
            lambda1 = 0.5 * (trace + discriminant)
            lambda2 = 0.5 * (trace - discriminant)
            
            # Ridge measure (Frangi-like)
            Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            # Ridge response
            beta = 0.5
            c = 0.5 * np.max(S)
            
            response = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*c**2)))
            response[lambda2 > 0] = 0  # Only negative λ2 for ridges
            
            # Scale normalization
            ridge_response = np.maximum(ridge_response, scale**2 * response)
            
        return ridge_response
    
    def _frangi_vesselness(self, image, scales=np.arange(1, 4, 0.5)):
        """
        Frangi vesselness filter for line-like structures.
        """
        vesselness = np.zeros_like(image)
        
        for scale in scales:
            # Gaussian derivatives
            smoothed = gaussian_filter(image, scale)
            
            # Hessian
            Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
            Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
            Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
            
            # Eigenvalues
            tmp = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
            lambda1 = 0.5 * (Hxx + Hyy + tmp)
            lambda2 = 0.5 * (Hxx + Hyy - tmp)
            
            # Sort eigenvalues by absolute value
            idx = np.abs(lambda1) < np.abs(lambda2)
            lambda1[idx], lambda2[idx] = lambda2[idx], lambda1[idx]
            
            # Vesselness measures
            Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
            S = np.sqrt(lambda1**2 + lambda2**2)
            
            # Parameters
            beta = 0.5
            gamma = 15
            
            # Vesselness
            v = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*gamma**2)))
            v[lambda2 > 0] = 0
            
            # Update maximum response
            vesselness = np.maximum(vesselness, v)
            
        return vesselness
    
    def _phase_congruency(self, image, nscale=4, norient=6):
        """
        Phase congruency for feature detection.
        """
        rows, cols = image.shape
        
        # Fourier transform
        IM = np.fft.fft2(image)
        
        # Initialize
        PC = np.zeros((rows, cols))
        
        # Frequency coordinates
        u, v = np.meshgrid(np.fft.fftfreq(cols), np.fft.fftfreq(rows))
        radius = np.sqrt(u**2 + v**2)
        radius[0, 0] = 1  # Avoid division by zero
        
        # Log-Gabor filters
        wavelength = 6
        for s in range(nscale):
            lambda_s = wavelength * (2**s)
            fo = 1.0 / lambda_s
            
            # Log-Gabor radial component
            logGabor = np.exp(-(np.log(radius/fo))**2 / (2 * np.log(0.65)**2))
            logGabor[radius < fo/3] = 0
            
            for o in range(norient):
                angle = o * np.pi / norient
                
                # Angular component
                theta = np.arctan2(v, u)
                ds = np.sin(theta - angle)
                dc = np.cos(theta - angle)
                spread = np.pi / norient / 1.5
                
                angular = np.exp(-(ds**2) / (2 * spread**2))
                
                # Combined filter
                filter_bank = logGabor * angular
                
                # Apply filter
                response = np.fft.ifft2(IM * filter_bank)
                
                # Phase congruency calculation
                magnitude = np.abs(response)
                phase = np.angle(response)
                
                PC += magnitude * np.cos(phase - np.mean(phase))
                
        # Normalize
        PC = PC / (nscale * norient)
        PC = (PC - PC.min()) / (PC.max() - PC.min() + 1e-10)
        
        return PC
    
    def _radon_line_detection(self, image):
        """
        Radon transform for line detection.
        """
        # Edge detection first
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        
        # Radon transform
        theta = np.linspace(0, 180, 180, endpoint=False)
        sinogram = self._radon_transform(edges, theta)
        
        # Find peaks in Radon space
        line_mask = np.zeros_like(image)
        
        # Threshold for peak detection
        threshold = np.mean(sinogram) + 2 * np.std(sinogram)
        
        # Back-project strong lines
        for i in range(sinogram.shape[1]):
            if np.max(sinogram[:, i]) > threshold:
                # Find peak
                rho_idx = np.argmax(sinogram[:, i])
                angle = theta[i] * np.pi / 180
                
                # Draw line
                self._draw_line_from_radon(line_mask, rho_idx - sinogram.shape[0]//2, 
                                          angle, image.shape)
                
        return line_mask
    
    def _radon_transform(self, image, theta):
        """
        Compute Radon transform.
        """
        # Pad image
        diagonal = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
        pad_x = (diagonal - image.shape[1]) // 2
        pad_y = (diagonal - image.shape[0]) // 2
        padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
        
        # Initialize sinogram
        sinogram = np.zeros((diagonal, len(theta)))
        
        # Compute projections
        for i, angle in enumerate(theta):
            rotated = ndimage.rotate(padded, angle, reshape=False, order=1)
            sinogram[:, i] = np.sum(rotated, axis=1)
            
        return sinogram

    def _draw_line_from_radon(self, mask, rho, theta, shape):
        """
        Draw line from Radon parameters.
        """
        h, w = shape
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Handle edge cases for nearly horizontal or vertical lines
        epsilon = 1e-10
        
        if abs(sin_t) < epsilon:
            # Nearly horizontal line (theta ≈ 0 or π)
            x = int(rho / cos_t) if abs(cos_t) > epsilon else 0
            if 0 <= x < w:
                mask[:, x] = 1
        elif abs(cos_t) < epsilon:
            # Nearly vertical line (theta ≈ π/2)
            y = int(rho / sin_t) if abs(sin_t) > epsilon else 0
            if 0 <= y < h:
                mask[y, :] = 1
        elif abs(cos_t) > abs(sin_t):
            # More horizontal than vertical
            for x in range(w):
                y_float = (rho - x * cos_t) / sin_t
                if not np.isfinite(y_float):
                    continue
                y = int(round(y_float))
                if 0 <= y < h:
                    mask[y, x] = 1
        else:
            # More vertical than horizontal
            for y in range(h):
                x_float = (rho - y * sin_t) / cos_t
                if not np.isfinite(x_float):
                    continue
                x = int(round(x_float))
                if 0 <= x < w:
                    mask[y, x] = 1

    def _directional_filter_bank(self, image, n_orientations=16):
        """
        Directional filter bank using steerable filters.
        """
        response_map = np.zeros_like(image)
        
        # Create oriented filters
        for i in range(n_orientations):
            angle = i * np.pi / n_orientations
            
            # Steerable filter coefficients
            kernel = self._create_steerable_filter(angle)
            
            # Apply filter
            response = cv2.filter2D(image, -1, kernel)
            
            # Non-maximum suppression along perpendicular direction
            suppressed = self._directional_nms(response, angle + np.pi/2)
            
            # Update maximum response
            response_map = np.maximum(response_map, suppressed)
            
        return response_map
    
    def _create_steerable_filter(self, angle, size=15, sigma=2):
        """
        Create steerable derivative filter.
        """
        x, y = np.meshgrid(np.arange(size) - size//2, 
                          np.arange(size) - size//2)
        
        # Rotate coordinates
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)
        
        # Second derivative of Gaussian
        g = np.exp(-(x_rot**2 + y_rot**2) / (2*sigma**2))
        kernel = -x_rot * g / (sigma**4)
        
        return kernel / np.sum(np.abs(kernel))
    
    def _directional_nms(self, response, angle):
        """
        Non-maximum suppression along direction.
        """
        h, w = response.shape
        suppressed = response.copy()
        
        # Direction vectors
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                # Interpolate along direction
                val1 = self._bilinear_interpolate(response, x + dx, y + dy)
                val2 = self._bilinear_interpolate(response, x - dx, y - dy)
                
                # Suppress if not maximum
                if response[y, x] < val1 or response[y, x] < val2:
                    suppressed[y, x] = 0
                    
        return suppressed
    
    def _bilinear_interpolate(self, img, x, y):
        """
        Bilinear interpolation.
        """
        x0 = int(np.floor(x))
        x1 = x0 + 1
        y0 = int(np.floor(y))
        y1 = y0 + 1
        
        # Boundary check
        x0 = np.clip(x0, 0, img.shape[1] - 1)
        x1 = np.clip(x1, 0, img.shape[1] - 1)
        y0 = np.clip(y0, 0, img.shape[0] - 1)
        y1 = np.clip(y1, 0, img.shape[0] - 1)
        
        # Interpolation weights
        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)
        
        return wa * img[y0, x0] + wb * img[y0, x1] + \
               wc * img[y1, x0] + wd * img[y1, x1]
    
    def _morphological_scratch_refinement(self, mask):
        """
        Morphological refinement for scratch-like structures.
        """
        # Create line structuring elements at different angles
        refined = np.zeros_like(mask)
        
        for angle in np.arange(0, 180, 15):
            # Create line SE
            length = 15
            se = self._create_line_se(length, angle)
            
            # Morphological closing to connect fragments
            closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se)
            
            # Opening to remove small objects
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se)
            
            refined = np.logical_or(refined, opened)
            
        return refined.astype(np.uint8)
    
    def _create_line_se(self, length, angle):
        """
        Create line structuring element.
        """
        angle_rad = angle * np.pi / 180
        
        # Create line coordinates
        x = np.arange(length) - length // 2
        y = np.zeros(length)
        
        # Rotate
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Convert to image coordinates
        x_rot = np.round(x_rot).astype(int)
        y_rot = np.round(y_rot).astype(int)
        
        # Create structuring element
        se_size = length + 2
        se = np.zeros((se_size, se_size), dtype=np.uint8)
        
        # Draw line
        for i in range(len(x_rot)):
            se[y_rot[i] + se_size//2, x_rot[i] + se_size//2] = 1
            
        return se
    
    def _detect_digs_multimethod(self, image):
        """
        Detect digs using multiple sophisticated methods.
        """
        methods = []
        
        # 1. Scale-normalized Laplacian of Gaussian
        log_blobs = self._scale_normalized_log(image)
        methods.append(log_blobs)
        
        # 2. Determinant of Hessian
        doh_blobs = self._determinant_of_hessian(image)
        methods.append(doh_blobs)
        
        # 3. MSER (Maximally Stable Extremal Regions)
        mser_blobs = self._mser_detection(image)
        methods.append(mser_blobs)
        
        # 4. Morphological reconstruction
        morph_blobs = self._morphological_blob_detection(image)
        methods.append(morph_blobs)
        
        # 5. Local binary patterns for texture
        lbp_blobs = self._lbp_blob_detection(image)
        methods.append(lbp_blobs)
        
        # Combine methods
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        combined = np.zeros_like(image)
        for method, weight in zip(methods, weights):
            combined += weight * method
            
        # Adaptive thresholding
        threshold = np.mean(combined) + 1.5 * np.std(combined)
        dig_mask = combined > threshold
        
        # Shape-based filtering
        dig_mask = self._shape_based_filtering(dig_mask)
        
        if self.debug:
            self.intermediate_results['dig_methods'] = methods
            self.intermediate_results['dig_combined'] = combined
            
        return dig_mask
    
    def _scale_normalized_log(self, image, scales=np.arange(2, 10, 1)):
        """
        Scale-normalized Laplacian of Gaussian.
        """
        blob_response = np.zeros_like(image)
        
        for scale in scales:
            # LoG filter
            log = ndimage.gaussian_laplace(image, scale)
            
            # Scale normalization
            log *= scale**2
            
            # Update maximum response
            blob_response = np.maximum(blob_response, np.abs(log))
            
        return blob_response
    
    def _determinant_of_hessian(self, image, scales=np.arange(2, 10, 1)):
        """
        Determinant of Hessian blob detection.
        """
        doh_response = np.zeros_like(image)
        
        for scale in scales:
            # Gaussian smoothing
            smoothed = gaussian_filter(image, scale)
            
            # Hessian components
            Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
            Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
            Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
            
            # Determinant
            det = Hxx * Hyy - Hxy**2
            
            # Scale normalization
            det *= scale**4
            
            # Update response
            doh_response = np.maximum(doh_response, np.abs(det))
            
        return doh_response
    
    def _mser_detection(self, image):
        """
        MSER blob detection.
        """
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # MSER detector
        mser = cv2.MSER_create(
            delta=5,
            min_area=10,
            max_area=1000,
            max_variation=0.25,
            min_diversity=0.2
        )
        
        # Detect regions
        regions, _ = mser.detectRegions(img_uint8)
        
        # Create mask
        mask = np.zeros_like(image)
        for region in regions:
            cv2.fillPoly(mask, [region], 1)
            
        return mask
    
    def _morphological_blob_detection(self, image):
        """
        Morphological reconstruction for blob detection.
        """
        # Multiple thresholds
        blob_mask = np.zeros_like(image)
        
        for h in np.linspace(0.1, 0.5, 5):
            # Regional maxima
            markers = image - h
            reconstruction = morphology.reconstruction(markers, image)
            regional_max = image - reconstruction
            
            # Threshold
            mask = regional_max > 0.01
            
            # Size filtering
            labeled = measure.label(mask)
            for region in measure.regionprops(labeled):
                if region.area > 5 and region.eccentricity < 0.8:
                    blob_mask[labeled == region.label] = 1
                    
        return blob_mask
    
    def _lbp_blob_detection(self, image):
        """
        Local Binary Patterns for texture-based blob detection.
        """
        # LBP computation
        radius = 3
        n_points = 8 * radius
        
        # Simplified LBP
        lbp = np.zeros_like(image)
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            
            # Shift image
            shifted = ndimage.shift(image, [dy, dx], order=1)
            
            # Compare with center
            lbp += (shifted > image) * (2**i)
            
        # LBP histogram features
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        
        # Anomaly detection based on histogram
        uniform_patterns = [0, 1, 3, 7, 15, 31, 63, 127, 255]  # Example uniform patterns
        anomaly_score = np.zeros_like(image)
        
        for y in range(radius, image.shape[0] - radius):
            for x in range(radius, image.shape[1] - radius):
                local_lbp = lbp[y-radius:y+radius+1, x-radius:x+radius+1]
                local_hist, _ = np.histogram(local_lbp, bins=256, range=(0, 256))
                
                # Compute anomaly score
                score = 0
                for pattern in uniform_patterns:
                    score += local_hist[pattern]
                    
                anomaly_score[y, x] = 1 - score / np.sum(local_hist)
                
        return anomaly_score
    
    def _shape_based_filtering(self, mask):
        """
        Filter blobs based on shape characteristics.
        """
        filtered = np.zeros_like(mask)
        
        # Label connected components
        labeled = measure.label(mask)
        
        for region in measure.regionprops(labeled):
            # Shape criteria for digs
            if (region.area > 10 and 
                region.eccentricity < 0.9 and  # Not too elongated
                region.solidity > 0.5):  # Not too irregular
                
                filtered[labeled == region.label] = 1
                
        return filtered
    
    def _refine_with_variational_methods(self, mask, image):
        """
        Refine detections using variational methods.
        """
        # Convert mask to float
        phi = mask.astype(np.float64)
        
        # Chan-Vese level set evolution
        refined = self._chan_vese_evolution(phi, image, iterations=50)
        
        # Threshold
        refined = refined > 0.5
        
        return refined
    
    def _chan_vese_evolution(self, phi, image, iterations=50, 
                            mu=0.1, lambda1=1, lambda2=1):
        """
        Chan-Vese level set method.
        """
        phi = phi.copy()
        
        for _ in range(iterations):
            # Compute region averages
            c1 = np.mean(image[phi > 0]) if np.any(phi > 0) else 0
            c2 = np.mean(image[phi <= 0]) if np.any(phi <= 0) else 0
            
            # Gradient and curvature
            phi_x = np.gradient(phi, axis=1)
            phi_y = np.gradient(phi, axis=0)
            
            # Curvature
            phi_xx = np.gradient(phi_x, axis=1)
            phi_yy = np.gradient(phi_y, axis=0)
            phi_xy = np.gradient(phi_x, axis=0)
            
            denominator = np.sqrt(phi_x**2 + phi_y**2 + 1e-8)
            curvature = (phi_xx * phi_y**2 - 2 * phi_x * phi_y * phi_xy + 
                        phi_yy * phi_x**2) / (denominator**3)
            
            # Evolution equation
            F = mu * curvature - lambda1 * (image - c1)**2 + lambda2 * (image - c2)**2
            
            # Update
            phi += 0.1 * F
            
        return phi
    
    def _reduce_false_positives_scratches(self, mask, image):
        """
        Advanced false positive reduction for scratches.
        """
        refined = mask.copy()
        
        # 1. Length filtering
        labeled = measure.label(refined)
        for region in measure.regionprops(labeled):
            # Compute skeleton
            skeleton = morphology.skeletonize(region.image)
            
            # Check length
            if np.sum(skeleton) < 20:  # Minimum length
                refined[labeled == region.label] = 0
                
        # 2. Linearity check using PCA
        labeled = measure.label(refined)
        for region in measure.regionprops(labeled):
            if region.area < 10:
                continue
                
            # Get coordinates
            coords = np.array(region.coords)
            
            # PCA
            centered = coords - np.mean(coords, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eig(cov)
            
            # Linearity measure
            linearity = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-10)
            
            if linearity < 0.8:  # Not linear enough
                refined[labeled == region.label] = 0
                
        # 3. Contrast validation
        labeled = measure.label(refined)
        for region in measure.regionprops(labeled):
            # Get intensity profile perpendicular to line
            if region.area < 10:
                continue
                
            # Sample along the major axis
            y0, x0 = region.centroid
            orientation = region.orientation
            
            # Perpendicular direction
            perp_angle = orientation + np.pi/2
            
            # Sample points
            contrast_values = []
            for d in range(-5, 6):
                y = int(y0 + d * np.sin(perp_angle))
                x = int(x0 + d * np.cos(perp_angle))
                
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    contrast_values.append(image[y, x])
                    
            if len(contrast_values) > 0:
                contrast = np.std(contrast_values)
                if contrast < 0.05:  # Low contrast
                    refined[labeled == region.label] = 0
                    
        return refined
    
    def _reduce_false_positives_digs(self, mask, image):
        """
        Advanced false positive reduction for digs.
        """
        refined = mask.copy()
        
        # 1. Circularity check
        labeled = measure.label(refined)
        for region in measure.regionprops(labeled):
            # Circularity
            circularity = 4 * np.pi * region.area / (region.perimeter**2 + 1e-10)
            
            if circularity < 0.4:  # Not circular enough
                refined[labeled == region.label] = 0
                
        # 2. Texture analysis
        labeled = measure.label(refined)
        for region in measure.regionprops(labeled):
            if region.area < 10:
                continue
                
            # Extract region
            min_row, min_col, max_row, max_col = region.bbox
            region_img = image[min_row:max_row, min_col:max_col]
            region_mask = region.image
            
            # Compute texture features
            masked_region = region_img * region_mask
            
            # Entropy
            entropy = -np.sum(masked_region * np.log(masked_region + 1e-10))
            
            # Contrast with surroundings
            dilated = morphology.dilation(region_mask, morphology.disk(3))
            boundary = dilated & ~region_mask
            
            if np.any(boundary):
                inside_mean = np.mean(masked_region[region_mask])
                outside_mean = np.mean(region_img[boundary])
                contrast = abs(inside_mean - outside_mean)
                
                if contrast < 0.05:  # Low contrast
                    refined[labeled == region.label] = 0
                    
        # 3. Size consistency
        labeled = measure.label(refined)
        areas = [r.area for r in measure.regionprops(labeled)]
        
        if len(areas) > 0:
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            
            for region in measure.regionprops(labeled):
                # Remove outliers
                if abs(region.area - mean_area) > 3 * std_area:
                    refined[labeled == region.label] = 0
                    
        return refined


def detect_fiber_defects(image_path):
    """
    Detect defects in fiber optic end face image.
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create detector
    detector = FiberOpticDefectDetector(debug=True)
    
    # Detect defects
    results = detector.detect_defects(image)
    
    # Visualize results
    visualize_results(image, results)
    
    return results

def visualize_results(image, results):
    """
    Visualize detection results.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Scratches
    axes[0, 1].imshow(results['scratches'], cmap='hot')
    axes[0, 1].set_title('Detected Scratches')
    axes[0, 1].axis('off')
    
    # Digs
    axes[0, 2].imshow(results['digs'], cmap='hot')
    axes[0, 2].set_title('Detected Digs')
    axes[0, 2].axis('off')
    
    # Combined
    axes[1, 0].imshow(results['combined'], cmap='hot')
    axes[1, 0].set_title('All Defects')
    axes[1, 0].axis('off')
    
    # Overlay on original
    overlay = np.stack([image, image, image], axis=-1)
    overlay[..., 0] = np.where(results['scratches'], 255, overlay[..., 0])
    overlay[..., 1] = np.where(results['digs'], 255, overlay[..., 1])
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (Red: Scratches, Green: Digs)')
    axes[1, 1].axis('off')
    
    # Statistics
    axes[1, 2].text(0.1, 0.8, f"Scratches: {np.sum(results['scratches'])} pixels", 
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f"Digs: {np.sum(results['digs'])} pixels", 
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, f"Total defects: {np.sum(results['combined'])} pixels", 
                    transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # usage
    results = detect_fiber_defects("/home/jarvis/Documents/GitHub/IPPS/image_batch/img38.jpg")
    pass
