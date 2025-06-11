#!/usr/bin/env python3
"""
Fiber Optic End Face Defect Detector
====================================
This script analyzes heatmap data to identify scratches (lines) and digs (dots)
in fiber optic end faces while ignoring the core and cladding rings.

Author: Assistant
Date: 2025
"""

import cv2
import numpy as np
import json
import h5py
from pathlib import Path
import argparse
import sys
from typing import Tuple, List, Dict, Any, Optional
import logging
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import morphology, measure, feature
from skimage.transform import probabilistic_hough_line
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FiberDefectDetector:
    """
    Detects scratches (lines) and digs (dots) in fiber optic end faces.
    """
    
    def __init__(self):
        """Initialize the detector."""
        self.difference_map = None
        self.intensity_matrix = None
        self.analysis_data = None
        self.detected_rings = []
        self.detected_scratches = []
        self.detected_digs = []
        self.mask_rings = None
        self.processed_map = None
        self.metadata = {}
        
    def load_heatmap_data(self, json_path: Optional[str] = None, 
                         image_path: Optional[str] = None,
                         npy_path: Optional[str] = None) -> None:
        """
        Load heatmap data from various sources.
        
        Args:
            json_path: Path to analysis JSON file
            image_path: Path to heatmap image
            npy_path: Path to difference map NPY file
        """
        # Load analysis JSON if provided
        if json_path and Path(json_path).exists():
            with open(json_path, 'r') as f:
                self.analysis_data = json.load(f)
            logger.info(f"Loaded analysis data from {json_path}")
        
        # Load difference map from NPY if provided
        if npy_path and Path(npy_path).exists():
            self.difference_map = np.load(npy_path).astype(np.float32)
            logger.info(f"Loaded difference map from {npy_path}")
        
        # Load heatmap image if provided
        if image_path and Path(image_path).exists():
            heatmap_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if heatmap_img is not None:
                # Convert to grayscale if needed
                if len(heatmap_img.shape) == 3:
                    # Extract red channel (assuming black to red heatmap)
                    self.processed_map = heatmap_img[:, :, 2].astype(np.float32)
                else:
                    self.processed_map = heatmap_img.astype(np.float32)
                logger.info(f"Loaded heatmap image from {image_path}")
        
        # Ensure we have at least one data source
        if self.difference_map is None and self.processed_map is None:
            raise ValueError("No valid data source loaded")
        
        # Use difference map as primary if available
        if self.difference_map is not None:
            self.processed_map = self.difference_map.copy()
        
        logger.info(f"Data shape: {self.processed_map.shape}")
    
    def detect_fiber_rings(self, num_rings: int = 2) -> List[Tuple[int, int, int]]:
        """
        Detect the circular rings (core and cladding) in the fiber.
        
        Args:
            num_rings: Expected number of rings (default: 2 for core and cladding)
            
        Returns:
            List of detected circles as (x, y, radius) tuples
        """
        # Create a binary image for circle detection
        _, binary = cv2.threshold(self.processed_map.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to enhance circles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Use multiple methods to detect circles
        circles_hough = []
        circles_contour = []
        
        # Method 1: Hough Circle Transform
        # Try multiple parameter sets
        param_sets = [
            {'dp': 1, 'minDist': 50, 'param1': 50, 'param2': 30},
            {'dp': 1.5, 'minDist': 100, 'param1': 100, 'param2': 50},
            {'dp': 2, 'minDist': 80, 'param1': 80, 'param2': 40}
        ]
        
        for params in param_sets:
            circles = cv2.HoughCircles(
                binary,
                cv2.HOUGH_GRADIENT,
                dp=params['dp'],
                minDist=params['minDist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=20,
                maxRadius=int(min(binary.shape) / 2)
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    circles_hough.append(tuple(circle))
        
        # Method 2: Contour-based circle detection
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                # Fit ellipse to contour
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    # Use average of major and minor axes as radius
                    radius = int((ellipse[1][0] + ellipse[1][1]) / 4)
                    
                    # Check if it's circular enough
                    circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                    if circularity > 0.7:  # Threshold for circularity
                        circles_contour.append((center[0], center[1], radius))
        
        # Method 3: Radial profile analysis
        height, width = self.processed_map.shape
        center_y, center_x = height // 2, width // 2
        
        # Compute radial average profile
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_radius = int(min(height, width) / 2)
        
        radial_profile = []
        for r in range(max_radius):
            mask = (distances >= r - 0.5) & (distances < r + 0.5)
            if mask.any():
                radial_profile.append(np.mean(self.processed_map[mask]))
            else:
                radial_profile.append(0)
        
        radial_profile = np.array(radial_profile)
        
        # Find peaks in radial profile (rings appear as peaks)
        peaks, properties = find_peaks(radial_profile, 
                                     height=np.mean(radial_profile),
                                     distance=20,
                                     prominence=np.std(radial_profile))
        
        circles_radial = [(center_x, center_y, int(peak)) for peak in peaks]
        
        # Combine all detected circles
        all_circles = circles_hough + circles_contour + circles_radial
        
        if not all_circles:
            logger.warning("No rings detected using standard methods")
            return []
        
        # Remove duplicates and cluster similar circles
        if len(all_circles) > 0:
            circles_array = np.array(all_circles)
            
            # Group circles by their centers to find concentric rings
            centers = circles_array[:, :2]
            
            # Find the most common center region
            clustering = DBSCAN(eps=50, min_samples=1).fit(centers)
            
            # Find the largest cluster (most likely the fiber center)
            unique_labels, counts = np.unique(clustering.labels_, return_counts=True)
            if len(unique_labels) > 0:
                main_cluster_label = unique_labels[np.argmax(counts)]
                
                # Get circles from the main cluster
                main_cluster_mask = clustering.labels_ == main_cluster_label
                main_cluster_circles = circles_array[main_cluster_mask]
                
                # Calculate the average center for the main cluster
                avg_center = np.mean(main_cluster_circles[:, :2], axis=0).astype(int)
                
                # Find concentric rings around this center
                concentric_rings = []
                for circle in main_cluster_circles:
                    # Check if the circle is close to the average center
                    center_dist = np.sqrt((circle[0] - avg_center[0])**2 + 
                                        (circle[1] - avg_center[1])**2)
                    if center_dist < 30:  # Within 30 pixels of average center
                        # Use the average center for all rings to ensure concentricity
                        concentric_rings.append((avg_center[0], avg_center[1], circle[2]))
                
                # Remove duplicate radii
                unique_rings = []
                used_radii = []
                for ring in sorted(concentric_rings, key=lambda x: x[2]):
                    radius = ring[2]
                    # Check if this radius is significantly different from existing ones
                    is_unique = True
                    for used_r in used_radii:
                        if abs(radius - used_r) < 15:  # Minimum radius difference
                            is_unique = False
                            break
                    
                    if is_unique:
                        unique_rings.append(ring)
                        used_radii.append(radius)
                
                # Sort by radius and select the most likely rings
                unique_rings.sort(key=lambda x: x[2])
                
                if len(unique_rings) >= num_rings:
                    self.detected_rings = unique_rings[:num_rings]
                else:
                    self.detected_rings = unique_rings
                    logger.warning(f"Only detected {len(unique_rings)} rings, expected {num_rings}")
            else:
                self.detected_rings = []
        
        logger.info(f"Detected {len(self.detected_rings)} fiber rings")
        for i, (x, y, r) in enumerate(self.detected_rings):
            logger.info(f"  Ring {i+1}: center=({x}, {y}), radius={r}")
        
        return self.detected_rings
    
    def create_ring_mask(self, expand_pixels: int = 5) -> np.ndarray:
        """
        Create a mask to exclude the detected rings from analysis.
        
        Args:
            expand_pixels: Number of pixels to expand the ring masks
            
        Returns:
            Binary mask (True where rings are NOT present)
        """
        height, width = self.processed_map.shape
        mask = np.ones((height, width), dtype=bool)
        
        # Create circular masks for each detected ring
        y, x = np.ogrid[:height, :width]
        
        for cx, cy, r in self.detected_rings:
            # Create annular mask (ring shape)
            inner_r = max(0, r - expand_pixels)
            outer_r = r + expand_pixels
            
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            ring_mask = (distance >= inner_r) & (distance <= outer_r)
            mask[ring_mask] = False
        
        self.mask_rings = mask
        logger.info(f"Created ring mask, {np.sum(mask)} pixels available for analysis")
        
        return mask
    
    def enhance_faint_defects(self, enhancement_factor: float = 2.0,
                            noise_threshold: float = 0.1) -> np.ndarray:
        """
        Enhance faint defects while suppressing noise.
        
        Args:
            enhancement_factor: Factor to enhance weak signals
            noise_threshold: Threshold below which to consider as noise
            
        Returns:
            Enhanced difference map
        """
        # Apply ring mask
        masked_map = self.processed_map.copy()
        if self.mask_rings is not None:
            masked_map[~self.mask_rings] = 0
        
        # Normalize to 0-1 range
        if masked_map.max() > 0:
            normalized = masked_map / masked_map.max()
        else:
            normalized = masked_map
        
        # Apply adaptive histogram equalization to enhance contrast
        # Convert to uint8 for CLAHE
        normalized_uint8 = (normalized * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized_uint8).astype(np.float32) / 255.0
        
        # Apply non-linear enhancement
        # Suppress noise, enhance weak signals
        enhanced = np.where(enhanced < noise_threshold, 0, enhanced)
        enhanced = np.power(enhanced, 1.0 / enhancement_factor)
        
        # Apply edge-preserving smoothing
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Re-apply mask
        if self.mask_rings is not None:
            enhanced[~self.mask_rings] = 0
        
        self.processed_map = enhanced
        logger.info("Enhanced faint defects")
        
        return enhanced
    
    def detect_scratches(self, min_length: int = 20, 
                        gap_tolerance: int = 5) -> List[Dict[str, Any]]:
        """
        Detect linear scratches using multiple methods.
        
        Args:
            min_length: Minimum length for a valid scratch
            gap_tolerance: Maximum gap to consider in line detection
            
        Returns:
            List of detected scratches with properties
        """
        # Prepare binary image for line detection
        _, binary = cv2.threshold(
            (self.processed_map * 255).astype(np.uint8), 
            30, 255, cv2.THRESH_BINARY
        )
        
        # Apply morphological operations to connect broken lines
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_line)
        
        scratches = []
        
        # Method 1: Probabilistic Hough Line Transform
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=10,
            minLineLength=min_length,
            maxLineGap=gap_tolerance
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                scratches.append({
                    'type': 'scratch',
                    'method': 'hough',
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': length,
                    'angle': angle,
                    'intensity': np.mean(self.processed_map[y1:y2+1, x1:x2+1])
                })
        
        # Method 2: Connected component analysis with shape criteria
        labeled = measure.label(binary, connectivity=2)
        regions = measure.regionprops(labeled, intensity_image=self.processed_map)
        
        for region in regions:
            # Check if region is elongated (likely a scratch)
            if region.area > min_length:
                # Calculate eccentricity (0 = circle, 1 = line)
                if region.eccentricity > 0.9:
                    # Get oriented bounding box
                    minr, minc, maxr, maxc = region.bbox
                    height = maxr - minr
                    width = maxc - minc
                    
                    # Calculate aspect ratio
                    aspect_ratio = max(height, width) / max(min(height, width), 1)
                    
                    if aspect_ratio > 3:  # Elongated shape
                        # Get endpoints of major axis
                        y0, x0 = region.centroid
                        orientation = region.orientation
                        
                        x1 = x0 + region.major_axis_length/2 * np.cos(orientation)
                        y1 = y0 - region.major_axis_length/2 * np.sin(orientation)
                        x2 = x0 - region.major_axis_length/2 * np.cos(orientation)
                        y2 = y0 + region.major_axis_length/2 * np.sin(orientation)
                        
                        scratches.append({
                            'type': 'scratch',
                            'method': 'component',
                            'start': (int(x1), int(y1)),
                            'end': (int(x2), int(y2)),
                            'length': region.major_axis_length,
                            'angle': orientation * 180 / np.pi,
                            'intensity': region.mean_intensity,
                            'area': region.area,
                            'eccentricity': region.eccentricity
                        })
        
        # Method 3: Ridge detection for faint scratches
        # Use Frangi filter to detect ridge-like structures
        try:
            from skimage.filters import frangi
        except ImportError:
            try:
                from skimage.filters.ridges import frangi
            except ImportError:
                logger.warning("Frangi filter not available, skipping ridge detection")
                frangi = None
        
        if frangi is not None:
            frangi_result = frangi(self.processed_map, sigmas=range(1, 4), black_ridges=False)
            _, frangi_binary = cv2.threshold(
                (frangi_result * 255).astype(np.uint8), 
                30, 255, cv2.THRESH_BINARY
            )
            
            # Apply skeleton to get line centers
            skeleton = morphology.skeletonize(frangi_binary > 0)
            
            # Find continuous paths in skeleton
            labeled_skeleton = measure.label(skeleton, connectivity=2)
            skeleton_regions = measure.regionprops(labeled_skeleton)
            
            for region in skeleton_regions:
                if region.area > min_length / 2:  # Skeleton is thinner
                    coords = region.coords
                    if len(coords) > 2:
                        # Fit line to coordinates
                        vx, vy, x, y = cv2.fitLine(
                            coords.astype(np.float32), 
                            cv2.DIST_L2, 0, 0.01, 0.01
                        )
                        
                        # Get line endpoints
                        t_values = [(c[1] - x) / vx[0] if vx[0] != 0 else 0 for c in coords]
                        t_min, t_max = min(t_values), max(t_values)
                        
                        x1 = int(x + t_min * vx[0])
                        y1 = int(y + t_min * vy[0])
                        x2 = int(x + t_max * vx[0])
                        y2 = int(y + t_max * vy[0])
                        
                        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        if length > min_length:
                            scratches.append({
                                'type': 'scratch',
                                'method': 'ridge',
                                'start': (x1, y1),
                                'end': (x2, y2),
                                'length': length,
                                'angle': np.arctan2(vy[0], vx[0]) * 180 / np.pi,
                                'intensity': np.mean([self.processed_map[c[0], c[1]] for c in coords])
                            })
        
        # Remove duplicate detections
        unique_scratches = []
        for scratch in scratches:
            is_duplicate = False
            for unique in unique_scratches:
                # Check if endpoints are close
                dist1 = np.sqrt((scratch['start'][0] - unique['start'][0])**2 + 
                               (scratch['start'][1] - unique['start'][1])**2)
                dist2 = np.sqrt((scratch['end'][0] - unique['end'][0])**2 + 
                               (scratch['end'][1] - unique['end'][1])**2)
                
                if dist1 < 10 and dist2 < 10:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_scratches.append(scratch)
        
        self.detected_scratches = unique_scratches
        logger.info(f"Detected {len(self.detected_scratches)} scratches")
        
        return self.detected_scratches
    
    def detect_digs(self, min_area: int = 10, 
                   max_area: int = 500,
                   circularity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Detect circular digs (dots) using multiple methods.
        
        Args:
            min_area: Minimum area for a valid dig
            max_area: Maximum area for a valid dig
            circularity_threshold: Minimum circularity (0-1)
            
        Returns:
            List of detected digs with properties
        """
        # Prepare binary image
        _, binary = cv2.threshold(
            (self.processed_map * 255).astype(np.uint8), 
            30, 255, cv2.THRESH_BINARY
        )
        
        # Apply morphological operations to fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        digs = []
        
        # Method 1: Blob detection
        # Invert for blob detector (it looks for dark blobs on light background)
        inverted = 255 - binary
        
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByCircularity = True
        params.minCircularity = circularity_threshold
        params.filterByConvexity = True
        params.minConvexity = 0.8
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(inverted)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            
            digs.append({
                'type': 'dig',
                'method': 'blob',
                'center': (x, y),
                'radius': radius,
                'area': np.pi * radius**2,
                'intensity': self.processed_map[y, x]
            })
        
        # Method 2: Connected components with circularity check
        labeled = measure.label(binary, connectivity=2)
        regions = measure.regionprops(labeled, intensity_image=self.processed_map)
        
        for region in regions:
            area = region.area
            if min_area <= area <= max_area:
                # Calculate circularity
                perimeter = region.perimeter
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    if circularity >= circularity_threshold:
                        # Check eccentricity (should be low for circles)
                        if region.eccentricity < 0.5:
                            y, x = region.centroid
                            equivalent_diameter = region.equivalent_diameter
                            
                            digs.append({
                                'type': 'dig',
                                'method': 'component',
                                'center': (int(x), int(y)),
                                'radius': int(equivalent_diameter / 2),
                                'area': area,
                                'intensity': region.mean_intensity,
                                'circularity': circularity,
                                'eccentricity': region.eccentricity
                            })
        
        # Method 3: Local maxima detection for very small digs
        # Apply distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find local maxima
        # Handle different versions of scikit-image
        try:
            # Newer versions return coordinates directly
            local_maxima = feature.peak_local_max(
                dist_transform,
                min_distance=5,
                threshold_abs=2
            )
        except TypeError:
            # Older versions might need different parameters
            coordinates = feature.peak_local_max(
                dist_transform,
                min_distance=5,
                threshold_abs=2,
                exclude_border=False
            )
            local_maxima = np.array(coordinates)
        
        for y, x in local_maxima:
            # Check if it's not part of a scratch
            is_scratch_part = False
            for scratch in self.detected_scratches:
                # Check distance to scratch line
                x1, y1 = scratch['start']
                x2, y2 = scratch['end']
                
                # Point to line distance
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if line_length > 0:
                    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length**2))
                    proj_x = x1 + t * (x2 - x1)
                    proj_y = y1 + t * (y2 - y1)
                    
                    dist_to_line = np.sqrt((x - proj_x)**2 + (y - proj_y)**2)
                    if dist_to_line < 5:
                        is_scratch_part = True
                        break
            
            if not is_scratch_part:
                radius = int(dist_transform[y, x])
                if radius >= 2:  # Minimum radius
                    digs.append({
                        'type': 'dig',
                        'method': 'maxima',
                        'center': (x, y),
                        'radius': radius,
                        'area': np.pi * radius**2,
                        'intensity': self.processed_map[y, x]
                    })
        
        # Remove duplicates
        unique_digs = []
        for dig in digs:
            is_duplicate = False
            for unique in unique_digs:
                dist = np.sqrt((dig['center'][0] - unique['center'][0])**2 + 
                              (dig['center'][1] - unique['center'][1])**2)
                if dist < 5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_digs.append(dig)
        
        self.detected_digs = unique_digs
        logger.info(f"Detected {len(self.detected_digs)} digs")
        
        return self.detected_digs
    
    def create_defect_visualization(self, 
                                  scratch_color: Tuple[int, int, int] = (255, 0, 0),
                                  dig_color: Tuple[int, int, int] = (255, 0, 0),
                                  ring_color: Tuple[int, int, int] = (0, 255, 0),
                                  background: str = 'black') -> np.ndarray:
        """
        Create visualization with detected defects highlighted in blue.
        
        Args:
            scratch_color: BGR color for scratches (default: blue)
            dig_color: BGR color for digs (default: blue)
            ring_color: BGR color for rings (default: green)
            background: 'black', 'white', or 'original'
            
        Returns:
            Visualization image
        """
        height, width = self.processed_map.shape
        
        # Create base image
        if background == 'black':
            vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        elif background == 'white':
            vis_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        else:  # original
            # Convert processed map to BGR
            vis_image = cv2.cvtColor(
                (self.processed_map * 255).astype(np.uint8), 
                cv2.COLOR_GRAY2BGR
            )
        
        # Draw rings (optional, in green)
        for x, y, r in self.detected_rings:
            cv2.circle(vis_image, (x, y), r, ring_color, 2)
        
        # Draw scratches in blue
        for scratch in self.detected_scratches:
            x1, y1 = scratch['start']
            x2, y2 = scratch['end']
            cv2.line(vis_image, (x1, y1), (x2, y2), scratch_color, 3)
        
        # Draw digs in blue
        for dig in self.detected_digs:
            x, y = dig['center']
            r = dig['radius']
            cv2.circle(vis_image, (x, y), max(r, 3), dig_color, -1)
        
        return vis_image
    
    def create_defect_matrix(self) -> np.ndarray:
        """
        Create a matrix where defects are marked with specific values.
        
        Returns:
            Matrix with 0=background, 1=rings, 2=scratches, 3=digs
        """
        height, width = self.processed_map.shape
        defect_matrix = np.zeros((height, width), dtype=np.uint8)
        
        # Mark rings
        y, x = np.ogrid[:height, :width]
        for cx, cy, r in self.detected_rings:
            ring_mask = np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r) <= 2
            defect_matrix[ring_mask] = 1
        
        # Mark scratches
        for scratch in self.detected_scratches:
            x1, y1 = scratch['start']
            x2, y2 = scratch['end']
            
            # Draw thick line
            cv2.line(defect_matrix, (x1, y1), (x2, y2), 2, 3)
        
        # Mark digs
        for dig in self.detected_digs:
            x, y = dig['center']
            r = max(dig['radius'], 2)
            cv2.circle(defect_matrix, (x, y), r, 3, -1)
        
        return defect_matrix
    
    def save_results(self, output_dir: str, base_name: str) -> None:
        """
        Save all detection results.
        
        Args:
            output_dir: Output directory
            base_name: Base filename for outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save visualization with blue defects
        vis_image = self.create_defect_visualization()
        vis_path = output_dir / f"{base_name}_defects_detected.png"
        cv2.imwrite(str(vis_path), vis_image)
        logger.info(f"Saved defect visualization to {vis_path}")
        
        # Save defect matrix
        defect_matrix = self.create_defect_matrix()
        matrix_path = output_dir / f"{base_name}_defect_matrix.npy"
        np.save(matrix_path, defect_matrix)
        logger.info(f"Saved defect matrix to {matrix_path}")
        
        # Save analysis JSON
        analysis = {
            'metadata': {
                'image_shape': self.processed_map.shape,
                'processing_date': str(Path.cwd()),
                'total_defects': len(self.detected_scratches) + len(self.detected_digs)
            },
            'rings': [
                {
                    'id': i,
                    'center_x': int(x),
                    'center_y': int(y),
                    'radius': int(r)
                }
                for i, (x, y, r) in enumerate(self.detected_rings)
            ],
            'scratches': [
                {
                    'id': i,
                    'start_x': int(s['start'][0]),
                    'start_y': int(s['start'][1]),
                    'end_x': int(s['end'][0]),
                    'end_y': int(s['end'][1]),
                    'length': float(s['length']),
                    'angle': float(s['angle']),
                    'intensity': float(s['intensity']),
                    'detection_method': s['method']
                }
                for i, s in enumerate(self.detected_scratches)
            ],
            'digs': [
                {
                    'id': i,
                    'center_x': int(d['center'][0]),
                    'center_y': int(d['center'][1]),
                    'radius': int(d['radius']),
                    'area': float(d['area']),
                    'intensity': float(d['intensity']),
                    'detection_method': d['method']
                }
                for i, d in enumerate(self.detected_digs)
            ],
            'defect_matrix': defect_matrix.tolist()
        }
        
        json_path = output_dir / f"{base_name}_defect_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved defect analysis to {json_path}")
        
        # Create comprehensive report visualization
        self.create_report_visualization(output_dir / f"{base_name}_report.png")
    
    def create_report_visualization(self, save_path: str) -> None:
        """
        Create a comprehensive report visualization.
        
        Args:
            save_path: Path to save the report
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. Original processed map
        im1 = axes[0].imshow(self.processed_map, cmap='hot')
        axes[0].set_title('Processed Difference Map')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # 2. Ring detection
        ring_vis = self.processed_map.copy()
        for x, y, r in self.detected_rings:
            cv2.circle(ring_vis, (x, y), r, 1, 2)
        axes[1].imshow(ring_vis, cmap='hot')
        axes[1].set_title(f'Detected Rings ({len(self.detected_rings)})')
        axes[1].axis('off')
        
        # 3. Enhanced map
        enhanced = self.enhance_faint_defects()
        im3 = axes[2].imshow(enhanced, cmap='hot')
        axes[2].set_title('Enhanced Defects')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        # 4. Scratch detection
        scratch_vis = np.zeros_like(self.processed_map)
        for scratch in self.detected_scratches:
            x1, y1 = scratch['start']
            x2, y2 = scratch['end']
            cv2.line(scratch_vis, (x1, y1), (x2, y2), 1, 2)
        axes[3].imshow(scratch_vis, cmap='Blues')
        axes[3].set_title(f'Detected Scratches ({len(self.detected_scratches)})')
        axes[3].axis('off')
        
        # 5. Dig detection
        dig_vis = np.zeros_like(self.processed_map)
        for dig in self.detected_digs:
            x, y = dig['center']
            r = max(dig['radius'], 2)
            cv2.circle(dig_vis, (x, y), r, 1, -1)
        axes[4].imshow(dig_vis, cmap='Blues')
        axes[4].set_title(f'Detected Digs ({len(self.detected_digs)})')
        axes[4].axis('off')
        
        # 6. Final result
        final_vis = self.create_defect_visualization(background='original')
        final_vis_rgb = cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB)
        axes[5].imshow(final_vis_rgb)
        axes[5].set_title('Final Defect Detection')
        axes[5].axis('off')
        
        plt.suptitle('Fiber Optic Defect Detection Report', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved report visualization to {save_path}")


def main():
    """Main function to run the fiber defect detector."""
    parser = argparse.ArgumentParser(
        description='Detect scratches and digs in fiber optic end faces'
    )
    parser.add_argument('--json', help='Path to analysis JSON file')
    parser.add_argument('--image', help='Path to heatmap image')
    parser.add_argument('--npy', help='Path to difference map NPY file')
    parser.add_argument('-o', '--output', help='Output directory', default='.')
    parser.add_argument('--rings', type=int, default=2, 
                       help='Expected number of rings (default: 2)')
    parser.add_argument('--min-scratch-length', type=int, default=20,
                       help='Minimum scratch length in pixels')
    parser.add_argument('--min-dig-area', type=int, default=10,
                       help='Minimum dig area in pixels')
    parser.add_argument('--enhancement', type=float, default=2.0,
                       help='Enhancement factor for faint defects')
    parser.add_argument('--no-rings', action='store_true',
                       help='Skip ring detection')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not any([args.json, args.image, args.npy]):
        parser.error("At least one input file (--json, --image, or --npy) is required")
    
    try:
        # Create detector
        detector = FiberDefectDetector()
        
        # Load data
        detector.load_heatmap_data(
            json_path=args.json,
            image_path=args.image,
            npy_path=args.npy
        )
        
        # Detect rings unless skipped
        if not args.no_rings:
            detector.detect_fiber_rings(num_rings=args.rings)
            detector.create_ring_mask()
        
        # Enhance faint defects
        detector.enhance_faint_defects(enhancement_factor=args.enhancement)
        
        # Detect defects
        scratches = detector.detect_scratches(min_length=args.min_scratch_length)
        digs = detector.detect_digs(min_area=args.min_dig_area)
        
        # Determine base name
        if args.json:
            base_name = Path(args.json).stem.replace('_analysis', '')
        elif args.image:
            base_name = Path(args.image).stem.replace('_heatmap', '')
        elif args.npy:
            base_name = Path(args.npy).stem.replace('_difference_map', '')
        else:
            base_name = 'fiber_defects'
        
        # Save results
        detector.save_results(args.output, base_name)
        
        # Print summary
        print("\n" + "="*50)
        print("FIBER DEFECT DETECTION SUMMARY")
        print("="*50)
        print(f"Detected rings: {len(detector.detected_rings)}")
        for i, (x, y, r) in enumerate(detector.detected_rings):
            print(f"  Ring {i+1}: center=({x}, {y}), radius={r}")
        print(f"\nDetected scratches: {len(scratches)}")
        for i, scratch in enumerate(scratches[:5]):  # Show first 5
            print(f"  Scratch {i+1}: length={scratch['length']:.1f}px, "
                  f"angle={scratch['angle']:.1f}°, method={scratch['method']}")
        if len(scratches) > 5:
            print(f"  ... and {len(scratches) - 5} more")
        print(f"\nDetected digs: {len(digs)}")
        for i, dig in enumerate(digs[:5]):  # Show first 5
            print(f"  Dig {i+1}: radius={dig['radius']}px, "
                  f"area={dig['area']:.1f}px², method={dig['method']}")
        if len(digs) > 5:
            print(f"  ... and {len(digs) - 5} more")
        print(f"\nTotal defects found: {len(scratches) + len(digs)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
