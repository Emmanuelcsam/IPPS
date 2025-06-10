#!/usr/bin/env python3
# advanced_scratch_detection.py

"""
Advanced Scratch Detection
=========================================
Based on yudarw's implementation with enhancements
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

class AdvancedScratchDetector:
    """
    Advanced scratch detection using multiple techniques
    """
    
    def __init__(self):
        self.methods = {
            'gradient': self._gradient_based_detection,
            'gabor': self._gabor_based_detection,
            'hessian': self._hessian_based_detection,
            'morphological': self._morphological_detection,
            'frequency': self._frequency_based_detection
        }
        
    def detect_scratches(self, image: np.ndarray, zone_mask: Optional[np.ndarray] = None,
                        methods: List[str] = ['gradient', 'gabor', 'hessian'],
                        fusion_weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Detect scratches using multiple methods
        
        Args:
            image: Input grayscale image
            zone_mask: Optional zone mask
            methods: List of detection methods to use
            fusion_weights: Weights for method fusion
            
        Returns:
            Final scratch mask and individual method results
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Default weights
        if fusion_weights is None:
            fusion_weights = {method: 1.0 for method in methods}
        
        # Apply zone mask if provided
        if zone_mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=zone_mask)
        else:
            masked_image = image
        
        # Run selected methods
        individual_results = {}
        combined_map = np.zeros_like(image, dtype=np.float32)
        
        for method in methods:
            if method in self.methods:
                try:
                    result = self.methods[method](masked_image)
                    individual_results[method] = result
                    
                    # Add weighted contribution
                    weight = fusion_weights.get(method, 1.0)
                    combined_map += result.astype(np.float32) * weight
                    
                except Exception as e:
                    logging.error(f"Error in {method} scratch detection: {e}")
                    continue
        
        # Normalize combined map
        if len(individual_results) > 0:
            combined_map /= sum(fusion_weights.values())
            combined_map = np.clip(combined_map, 0, 255).astype(np.uint8)
        
        # Apply zone mask to final result
        if zone_mask is not None:
            combined_map = cv2.bitwise_and(combined_map, combined_map, mask=zone_mask)
        
        # Post-processing
        final_mask = self._postprocess_scratch_mask(combined_map)
        
        return final_mask, individual_results
    
    def _gradient_based_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Gradient-based scratch detection
        """
        # Multi-scale gradient analysis
        scales = [1, 2, 3]
        gradient_responses = []
        
        for scale in scales:
            # Gaussian smoothing at current scale
            sigma = scale * 0.5
            smoothed = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # Compute gradients
            grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Gradient direction
            grad_dir = np.arctan2(grad_y, grad_x)
            
            # Directional non-maximum suppression
            nms_result = self._directional_nms(grad_mag, grad_dir)
            
            gradient_responses.append(nms_result)
        
        # Combine multi-scale responses
        combined = np.max(gradient_responses, axis=0)
        
        # Normalize
        combined = cv2.normalize(combined, combined, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Threshold
        _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _gabor_based_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Gabor filter based scratch detection
        """
        # Gabor parameters optimized for scratches
        ksize = 21
        sigma = 3.0
        lambd = 10.0
        gamma = 0.5
        psi = 0
        
        # Multiple orientations
        orientations = np.arange(0, np.pi, np.pi / 8)
        
        gabor_responses = []
        
        for theta in orientations:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
            
            # Apply filter
            filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
            
            # Take absolute value
            response = np.abs(filtered)
            
            gabor_responses.append(response)
        
        # Max response across orientations
        max_response = np.max(gabor_responses, axis=0)
        
        # Enhance linear structures
        # Apply morphological operations with oriented kernels
        enhanced = np.zeros_like(max_response)
        
        for theta in orientations:
            # Create oriented kernel
            kernel_length = 15
            kernel = self._create_oriented_kernel(kernel_length, theta, width=3)
            
            # Morphological closing
            closed = cv2.morphologyEx(max_response.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            enhanced = np.maximum(enhanced, closed)
        
        # Normalize and threshold
        enhanced = cv2.normalize(enhanced, enhanced, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _hessian_based_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Hessian matrix based scratch detection
        """
        # Multi-scale Hessian analysis
        scales = [1.0, 1.5, 2.0]
        hessian_responses = []
        
        for sigma in scales:
            try:
                # Compute Hessian matrix components
                Hxx, Hxy, Hyy = hessian_matrix(image, sigma=int(sigma), order='xy')
                
                # Compute eigenvalues correctly - create the full Hessian matrix
                # hessian_matrix_eigvals expects a single Hessian tensor
                h, w = image.shape
                hessian_tensor = np.zeros((h, w, 2, 2))
                hessian_tensor[:, :, 0, 0] = Hxx
                hessian_tensor[:, :, 0, 1] = Hxy
                hessian_tensor[:, :, 1, 0] = Hxy
                hessian_tensor[:, :, 1, 1] = Hyy
                
                # Now call with the correct format
                eigenvalues = hessian_matrix_eigvals(hessian_tensor)
                lambda1 = eigenvalues[0]
                lambda2 = eigenvalues[1]
                
                # Ridge measure (scratches are ridges)
                beta = 0.5
                c = 15
                
                # Ensure lambda2 is the larger eigenvalue
                lambda1_abs = np.abs(lambda1)
                lambda2_abs = np.abs(lambda2)
                
                # Ridge measure
                Rb = lambda1_abs / (lambda2_abs + 1e-10)
                S = np.sqrt(lambda1_abs**2 + lambda2_abs**2)
                
                # Vesselness measure
                vesselness = np.exp(-(Rb**2) / (2 * beta**2)) * (1 - np.exp(-(S**2) / (2 * c**2)))
                
                # Only keep where lambda2 < 0 (bright ridges on dark background)
                vesselness[lambda2 >= 0] = 0
                
                hessian_responses.append(vesselness)
            except Exception as e:
                logging.error(f"Error in hessian scratch detection: {e}")
                # Return zeros on error
                hessian_responses.append(np.zeros_like(image))
        
        # Combine scales
        combined = np.max(hessian_responses, axis=0) if hessian_responses else np.zeros_like(image)
        
        # Normalize
        combined = cv2.normalize(combined, combined, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Threshold
        _, binary = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def _morphological_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Morphological approach for scratch detection
        """
        # Multiple oriented top-hat transforms
        orientations = np.arange(0, 180, 15)
        tophat_responses = []
        
        for angle in orientations:
            # Create oriented structuring element
            kernel_length = 21
            kernel = self._create_oriented_kernel(kernel_length, np.deg2rad(angle), width=3)
            
            # Black top-hat (for dark scratches)
            blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            
            # White top-hat (for bright scratches)
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            
            # Combine both
            combined = cv2.add(blackhat, tophat)
            tophat_responses.append(combined)
        
        # Max response
        max_response = np.max(tophat_responses, axis=0)
        
        # Enhance contrast
        enhanced = cv2.normalize(enhanced, enhanced, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        return binary
    
    def _frequency_based_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Frequency domain analysis for scratch detection
        """
        # Apply FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create directional filters in frequency domain
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Multiple oriented band-pass filters
        orientations = np.arange(0, 180, 15)
        filtered_images = []
        
        for angle in orientations:
            # Create oriented band-pass filter
            mask = self._create_oriented_bandpass(rows, cols, angle, width=10)
            
            # Apply filter
            filtered_fshift = f_shift * mask
            
            # Inverse FFT
            filtered_ishift = np.fft.ifftshift(filtered_fshift)
            filtered_image = np.fft.ifft2(filtered_ishift)
            filtered_image = np.abs(filtered_image)
            
            filtered_images.append(filtered_image)
        
        # Combine filtered images
        combined = np.max(filtered_images, axis=0)
        
        # Normalize and threshold
        combined = cv2.normalize(combined, combined, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _create_oriented_kernel(self, length: int, angle: float, width: int = 3) -> np.ndarray:
        """
        Create an oriented linear kernel
        """
        # Create a horizontal line kernel
        kernel = np.zeros((length, length), dtype=np.uint8)
        center = length // 2
        
        # Draw horizontal line
        kernel[center - width//2:center + width//2 + 1, :] = 1
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((center, center), np.degrees(angle), 1)
        rotated = cv2.warpAffine(kernel, M, (length, length))
        
        return rotated
    
    def _create_oriented_bandpass(self, rows: int, cols: int, angle: float, width: int) -> np.ndarray:
        """
        Create oriented band-pass filter in frequency domain
        """
        # Create coordinate grids
        u = np.arange(cols) - cols // 2
        v = np.arange(rows) - rows // 2
        U, V = np.meshgrid(u, v)
        
        # Rotate coordinates
        angle_rad = np.deg2rad(angle)
        U_rot = U * np.cos(angle_rad) + V * np.sin(angle_rad)
        V_rot = -U * np.sin(angle_rad) + V * np.cos(angle_rad)
        
        # Create band-pass filter
        mask = np.exp(-(V_rot**2) / (2 * width**2))
        
        # Suppress DC component
        center_mask = np.sqrt(U**2 + V**2) > 5
        mask = mask * center_mask
        
        return mask
    
    def _directional_nms(self, magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Directional non-maximum suppression
        """
        rows, cols = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                q = 255
                r = 255
                
                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j+1]
                    r = magnitude[i, j-1]
                # Angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i+1, j-1]
                    r = magnitude[i-1, j+1]
                # Angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i+1, j]
                    r = magnitude[i-1, j]
                # Angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i-1, j-1]
                    r = magnitude[i+1, j+1]
                
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]
        
        return suppressed
    
    def _postprocess_scratch_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Post-process scratch detection results
        """
        # Remove small components
        min_scratch_length = 20
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        cleaned_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            # Get component properties
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate aspect ratio
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            
            # Keep if it looks like a scratch (elongated)
            if aspect_ratio > 3 and max(width, height) > min_scratch_length:
                cleaned_mask[labels == i] = 255
        
        # Apply skeletonization to thin scratches
        skeleton = skeletonize(cleaned_mask > 0).astype(np.uint8) * 255
        
        # Dilate slightly to make visible
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        final_mask = cv2.dilate(skeleton, kernel, iterations=1)
        
        return final_mask