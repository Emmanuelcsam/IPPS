#!/usr/bin/env python3
# performance_optimizer.py

"""
Performance Optimization Module
================================
Provides optimized image processing functions and utilities for improving
the speed of fiber optic inspection operations.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from functools import wraps
import time

def performance_timer(func):
    """Decorator to measure and log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"{func.__name__} took {elapsed_time:.2f} seconds")
        return result
    return wrapper

class ImageProcessor:
    """
    Optimized image processor with various performance enhancements
    """
    
    def __init__(self, max_processing_size: int = 1024, enable_gpu: bool = False):
        """
        Initialize the image processor
        
        Args:
            max_processing_size: Maximum dimension for processing (images larger than this will be resized)
            enable_gpu: Whether to attempt to use GPU acceleration (requires opencv built with CUDA)
        """
        self.max_processing_size = max_processing_size
        self.enable_gpu = enable_gpu and self._check_gpu_availability()
        
        if self.enable_gpu:
            logging.info("GPU acceleration enabled for image processing")
        else:
            logging.debug("Using CPU for image processing")
    
    def _check_gpu_availability(self) -> bool:
        """Check if OpenCV has CUDA support"""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False
    
    def resize_for_processing(self, image: np.ndarray, max_size: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Resize image for faster processing if needed.
        
        Args:
            image: Input image
            max_size: Maximum dimension for processing (uses instance default if None)
            
        Returns:
            Tuple of (processed_image, scale_factor)
        """
        if max_size is None:
            max_size = self.max_processing_size
            
        h, w = image.shape[:2]
        scale_factor = 1.0
        
        if max(h, w) > max_size:
            scale_factor = max_size / max(h, w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            # Use INTER_AREA for downscaling (best quality)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logging.debug(f"Image resized from {w}x{h} to {new_w}x{new_h} (scale: {scale_factor:.2f})")
            return resized, scale_factor
        
        return image, scale_factor
    
    @performance_timer
    def adaptive_resize(self, image: np.ndarray, target_area: int = 1000000) -> Tuple[np.ndarray, float]:
        """
        Adaptively resize image based on total pixel count rather than max dimension
        
        Args:
            image: Input image
            target_area: Target number of pixels (default 1 megapixel)
            
        Returns:
            Tuple of (processed_image, scale_factor)
        """
        h, w = image.shape[:2]
        current_area = h * w
        
        if current_area > target_area:
            scale_factor = np.sqrt(target_area / current_area)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logging.debug(f"Adaptive resize: {w}x{h} ({current_area/1e6:.1f}MP) -> {new_w}x{new_h} ({target_area/1e6:.1f}MP)")
            return resized, scale_factor
        
        return image, 1.0
    
    def optimize_preprocessing(self, image: np.ndarray, profile_config: Dict[str, Any]) -> np.ndarray:
        """
        Optimized preprocessing pipeline
        
        Args:
            image: Input grayscale image
            profile_config: Processing profile configuration
            
        Returns:
            Preprocessed image
        """
        # Get preprocessing parameters
        preproc_params = profile_config.get("preprocessing", {})
        
        # Apply CLAHE with optimized parameters
        clahe_clip = preproc_params.get("clahe_clip_limit", 2.0)
        clahe_grid = tuple(preproc_params.get("clahe_tile_grid_size", [8, 8]))
        
        if self.enable_gpu:
            # GPU-accelerated CLAHE if available
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            clahe = cv2.cuda.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
            gpu_result = clahe.apply(gpu_image)
            result = gpu_result.download()
        else:
            # CPU CLAHE
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
            result = clahe.apply(image)
        
        # Apply Gaussian blur
        blur_kernel = tuple(preproc_params.get("gaussian_blur_kernel_size", [5, 5]))
        blur_kernel = tuple(k if k % 2 == 1 else k + 1 for k in blur_kernel)  # Ensure odd
        
        if self.enable_gpu and blur_kernel[0] <= 31:  # GPU has kernel size limits
            gpu_result = cv2.cuda_GpuMat()
            gpu_result.upload(result)
            gpu_blurred = cv2.cuda.bilateralFilter(gpu_result, -1, 50, 50)
            result = gpu_blurred.download()
        else:
            result = cv2.GaussianBlur(result, blur_kernel, 0)
        
        return result
    
    @performance_timer
    def fast_defect_detection(self, image: np.ndarray, mask: np.ndarray, 
                            threshold_params: Dict[str, Any]) -> np.ndarray:
        """
        Fast defect detection using optimized algorithms
        
        Args:
            image: Preprocessed image
            mask: Zone mask
            threshold_params: Threshold parameters
            
        Returns:
            Binary defect mask
        """
        # Apply mask efficiently
        masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        
        # Adaptive thresholding with optimized block size
        block_size = threshold_params.get("block_size", 11)
        C = threshold_params.get("C", 2)
        
        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        binary = cv2.adaptiveThreshold(
            masked_image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, C
        )
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Remove small objects efficiently
        min_area = threshold_params.get("min_defect_area", 10)
        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Create mask of components to keep
            keep_mask = np.zeros_like(binary)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    keep_mask[labels == i] = 255
            
            binary = keep_mask
        
        return binary
    
    def batch_process_images(self, image_paths: list, process_func: callable, 
                           batch_size: int = 4) -> list:
        """
        Process images in batches for better memory efficiency
        
        Args:
            image_paths: List of image paths
            process_func: Function to process each image
            batch_size: Number of images to process simultaneously
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batch_results = []
            
            # Process batch
            for path in batch:
                try:
                    result = process_func(path)
                    batch_results.append(result)
                except Exception as e:
                    logging.error(f"Error processing {path}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
            
            # Clear memory between batches
            import gc
            gc.collect()
        
        return results

class MemoryOptimizer:
    """
    Memory optimization utilities
    """
    
    @staticmethod
    def estimate_memory_usage(image_shape: Tuple[int, int], num_channels: int = 1) -> int:
        """
        Estimate memory usage for an image
        
        Args:
            image_shape: (height, width) of the image
            num_channels: Number of channels
            
        Returns:
            Estimated memory usage in bytes
        """
        h, w = image_shape
        # Assume float32 for processing
        bytes_per_pixel = 4 * num_channels
        return h * w * bytes_per_pixel
    
    @staticmethod
    def get_optimal_batch_size(image_shape: Tuple[int, int], available_memory_gb: float = 4.0) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            image_shape: (height, width) of images
            available_memory_gb: Available memory in GB
            
        Returns:
            Optimal batch size
        """
        image_memory = MemoryOptimizer.estimate_memory_usage(image_shape)
        # Reserve 50% of memory for processing overhead
        usable_memory = available_memory_gb * 1e9 * 0.5
        batch_size = int(usable_memory / image_memory)
        return max(1, min(batch_size, 16))  # Cap at 16 for practical reasons

def optimize_hough_circles(image: np.ndarray, min_radius: int, max_radius: int,
                          sensitivity: float = 1.0) -> Optional[np.ndarray]:
    """
    Optimized Hough circles detection with multi-scale approach
    
    Args:
        image: Input image
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        sensitivity: Detection sensitivity (lower = more sensitive)
        
    Returns:
        Detected circles or None
    """
    # Multi-scale detection for better accuracy
    scales = [1.0, 0.8, 1.2]
    all_circles = []
    
    for scale in scales:
        scaled_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        circles = cv2.HoughCircles(
            scaled_img,
            cv2.HOUGH_GRADIENT,
            dp=1.2 * sensitivity,
            minDist=int(min(scaled_img.shape) * 0.1),
            param1=70,
            param2=30 * sensitivity,
            minRadius=int(min_radius * scale),
            maxRadius=int(max_radius * scale)
        )
        
        if circles is not None:
            # Scale circles back to original size
            circles = circles[0] / scale
            all_circles.extend(circles)
    
    if all_circles:
        # Remove duplicate detections
        all_circles = np.array(all_circles)
        # Simple clustering to merge nearby circles
        unique_circles = []
        for circle in all_circles:
            if not unique_circles or all(np.linalg.norm(circle[:2] - c[:2]) > 10 for c in unique_circles):
                unique_circles.append(circle)
        
        return np.array([unique_circles]) if unique_circles else None
    
    return None

# Cache for frequently used operations
class ProcessingCache:
    """Simple cache for expensive operations"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        return self.cache.get(key)
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest item (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()

# Global cache instance
_processing_cache = ProcessingCache()

def cached_operation(cache_key_func):
    """Decorator for caching expensive operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache_key_func(*args, **kwargs)
            cached_result = _processing_cache.get(cache_key)
            
            if cached_result is not None:
                logging.debug(f"Cache hit for {func.__name__} with key {cache_key}")
                return cached_result
            
            result = func(*args, **kwargs)
            _processing_cache.put(cache_key, result)
            return result
        
        return wrapper
    return decorator