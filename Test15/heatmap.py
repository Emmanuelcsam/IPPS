#!/usr/bin/env python3
"""
Intensity Difference Heatmap Generator
=====================================
This script analyzes pixel intensity matrices and creates heatmap images
showing where large differences occur between neighboring pixels.

Author: Assistant
Date: 2025
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import json
import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
from scipy import ndimage
from skimage import filters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntensityDifferenceAnalyzer:
    """
    Analyzes intensity matrices to find and visualize local differences.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.intensity_matrix = None
        self.difference_map = None
        self.metadata = {}
        self.source_file = None
        
    def load_intensity_matrix(self, file_path: str) -> np.ndarray:
        """
        Load intensity matrix from various file formats.
        
        Args:
            file_path: Path to the intensity matrix file
            
        Returns:
            Loaded intensity matrix as numpy array
        """
        file_path = Path(file_path)
        self.source_file = file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load accordingly
        suffix = file_path.suffix.lower()
        
        if suffix == '.npy':
            self.intensity_matrix = np.load(file_path).astype(np.float32)
            logger.info(f"Loaded NumPy array from {file_path}")
            
        elif suffix == '.csv':
            # Try to determine CSV format
            try:
                # First, try to read as simple matrix
                self.intensity_matrix = np.loadtxt(file_path, delimiter=',').astype(np.float32)
                logger.info(f"Loaded CSV matrix from {file_path}")
            except:
                # If that fails, try reading with coordinates
                df = pd.read_csv(file_path)
                if all(col in df.columns for col in ['x', 'y', 'intensity']):
                    # Reconstruct matrix from coordinates
                    max_x = df['x'].max() + 1
                    max_y = df['y'].max() + 1
                    self.intensity_matrix = np.zeros((max_y, max_x), dtype=np.float32)
                    
                    for _, row in df.iterrows():
                        self.intensity_matrix[int(row['y']), int(row['x'])] = row['intensity']
                    
                    logger.info(f"Loaded CSV with coordinates from {file_path}")
                else:
                    raise ValueError("CSV format not recognized")
                    
        elif suffix in ['.h5', '.hdf5']:
            with h5py.File(file_path, 'r') as f:
                # Find the intensity matrix dataset
                if 'intensity_matrix' in f:
                    self.intensity_matrix = f['intensity_matrix'][:].astype(np.float32)
                    # Load metadata if available
                    for key, value in f['intensity_matrix'].attrs.items():
                        self.metadata[key] = value
                else:
                    # Try to find any 2D dataset
                    for key in f.keys():
                        if len(f[key].shape) == 2:
                            self.intensity_matrix = f[key][:].astype(np.float32)
                            break
                    if self.intensity_matrix is None:
                        raise ValueError("No 2D dataset found in HDF5 file")
            logger.info(f"Loaded HDF5 file from {file_path}")
            
        elif suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'intensity_matrix' in data:
                self.intensity_matrix = np.array(data['intensity_matrix'], dtype=np.float32)
                if 'metadata' in data:
                    self.metadata = data['metadata']
            elif isinstance(data, list):
                self.intensity_matrix = np.array(data, dtype=np.float32)
            else:
                raise ValueError("JSON format not recognized")
            logger.info(f"Loaded JSON file from {file_path}")
            
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Validate matrix
        if len(self.intensity_matrix.shape) != 2:
            raise ValueError(f"Expected 2D matrix, got shape: {self.intensity_matrix.shape}")
        
        logger.info(f"Matrix shape: {self.intensity_matrix.shape}")
        logger.info(f"Value range: [{self.intensity_matrix.min():.2f}, {self.intensity_matrix.max():.2f}]")
        
        return self.intensity_matrix
    
    def calculate_differences(self, method: str = 'gradient_magnitude', 
                            neighborhood: str = '8-connected',
                            normalize: bool = True) -> np.ndarray:
        """
        Calculate local differences in the intensity matrix.
        
        Args:
            method: Method for calculating differences
                   - 'gradient_magnitude': Combined gradient magnitude
                   - 'max_neighbor': Maximum difference to any neighbor
                   - 'sobel': Sobel edge detection
                   - 'laplacian': Laplacian edge detection
                   - 'canny_strength': Canny edge strength
            neighborhood: Type of neighborhood to consider
                   - '4-connected': Only horizontal and vertical neighbors
                   - '8-connected': All 8 surrounding pixels
            normalize: Whether to normalize the result to 0-255 range
            
        Returns:
            Difference map as numpy array
        """
        if self.intensity_matrix is None:
            raise ValueError("No intensity matrix loaded")
        
        height, width = self.intensity_matrix.shape
        
        if method == 'gradient_magnitude':
            # Calculate gradients in x and y directions
            grad_x = np.zeros_like(self.intensity_matrix)
            grad_y = np.zeros_like(self.intensity_matrix)
            
            # Compute gradients with proper boundary handling
            grad_x[:, 1:] = self.intensity_matrix[:, 1:] - self.intensity_matrix[:, :-1]
            grad_y[1:, :] = self.intensity_matrix[1:, :] - self.intensity_matrix[:-1, :]
            
            # Gradient magnitude
            self.difference_map = np.sqrt(grad_x**2 + grad_y**2)
            
        elif method == 'max_neighbor':
            # Calculate maximum difference to any neighbor
            self.difference_map = np.zeros_like(self.intensity_matrix)
            
            # Define neighbor offsets based on connectivity
            if neighborhood == '4-connected':
                offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            else:  # 8-connected
                offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
            
            # Pad the array to handle boundaries
            padded = np.pad(self.intensity_matrix, 1, mode='edge')
            
            for dy, dx in offsets:
                neighbor = padded[1+dy:height+1+dy, 1+dx:width+1+dx]
                diff = np.abs(self.intensity_matrix - neighbor)
                self.difference_map = np.maximum(self.difference_map, diff)
                
        elif method == 'sobel':
            # Sobel edge detection
            sobel_x = cv2.Sobel(self.intensity_matrix, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(self.intensity_matrix, cv2.CV_64F, 0, 1, ksize=3)
            self.difference_map = np.sqrt(sobel_x**2 + sobel_y**2)
            
        elif method == 'laplacian':
            # Laplacian edge detection
            self.difference_map = np.abs(cv2.Laplacian(self.intensity_matrix, cv2.CV_64F))
            
        elif method == 'canny_strength':
            # Use Canny-like gradient calculation
            # Gaussian blur first
            blurred = cv2.GaussianBlur(self.intensity_matrix, (5, 5), 1.4)
            
            # Calculate gradients
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            self.difference_map = np.sqrt(grad_x**2 + grad_y**2)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize if requested
        if normalize and self.difference_map.max() > 0:
            self.difference_map = (self.difference_map / self.difference_map.max() * 255).astype(np.float32)
        
        logger.info(f"Calculated differences using {method} method")
        logger.info(f"Difference range: [{self.difference_map.min():.2f}, {self.difference_map.max():.2f}]")
        
        return self.difference_map
    
    def create_heatmap(self, 
                      threshold: float = 0.0,
                      color_map: str = 'black_to_red',
                      gamma: float = 1.0,
                      blur_radius: Optional[int] = None,
                      highlight_all_nonzero: bool = False) -> np.ndarray:
        """
        Create a heatmap image from the difference map.
        
        Args:
            threshold: Minimum difference value to show (0-255)
            color_map: Color mapping scheme
                      - 'black_to_red': Black (no diff) to bright red (high diff)
                      - 'black_red_yellow': Black to red to yellow
                      - 'heat': Traditional heat map
                      - 'custom': Custom gradient
            gamma: Gamma correction for emphasis (< 1 = emphasize low values, > 1 = emphasize high values)
            blur_radius: Optional Gaussian blur radius for smoothing
            highlight_all_nonzero: If True, highlight all non-zero areas with enhanced visibility
            
        Returns:
            Heatmap image as BGR numpy array
        """
        if self.difference_map is None:
            raise ValueError("No difference map calculated")
        
        # Apply threshold
        diff_map = self.difference_map.copy()
        diff_map[diff_map < threshold] = 0
        
        # Special handling for highlight_all_nonzero mode
        if highlight_all_nonzero:
            # Create a mask of all non-zero areas
            nonzero_mask = self.difference_map > 0
            
            # Enhance visibility of faint areas
            # Set a minimum visible value for all non-zero pixels
            min_visible_value = 30  # Minimum brightness to ensure visibility
            
            # Scale the differences but ensure minimum visibility
            if diff_map.max() > 0:
                diff_map = (diff_map / diff_map.max() * 255).astype(np.float32)
                # Ensure all non-zero values are at least min_visible_value
                diff_map[nonzero_mask & (diff_map < min_visible_value)] = min_visible_value
            
            # Apply special gamma correction to enhance faint areas
            if gamma == 1.0:
                gamma = 0.5  # Default to emphasizing low values in this mode
        
        # Apply gamma correction
        if gamma != 1.0:
            diff_map = np.power(diff_map / 255.0, gamma) * 255.0
        
        # Apply blur if requested
        if blur_radius and blur_radius > 0:
            diff_map = cv2.GaussianBlur(diff_map, (blur_radius*2+1, blur_radius*2+1), 0)
        
        # Normalize to 0-255 range
        if diff_map.max() > 0:
            diff_map = (diff_map / diff_map.max() * 255).astype(np.uint8)
        else:
            diff_map = diff_map.astype(np.uint8)
        
        # Create color map
        if color_map == 'black_to_red':
            # Pure black to pure red
            heatmap = np.zeros((*diff_map.shape, 3), dtype=np.uint8)
            heatmap[:, :, 2] = diff_map  # Red channel (BGR format)
            
        elif color_map == 'black_red_yellow':
            # Black -> Red -> Yellow
            heatmap = np.zeros((*diff_map.shape, 3), dtype=np.uint8)
            
            # Red channel: increases linearly
            heatmap[:, :, 2] = diff_map
            
            # Green channel: increases after halfway point (creates yellow)
            halfway = 128
            mask = diff_map > halfway
            heatmap[mask, 1] = ((diff_map[mask] - halfway) * 2).astype(np.uint8)
            
        elif color_map == 'heat':
            # Use OpenCV's heat colormap
            heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_HOT)
            
        elif color_map == 'highlight':
            # Special highlight mode for maximum visibility of all non-zero areas
            heatmap = np.zeros((*diff_map.shape, 3), dtype=np.uint8)
            
            # Create distinct colors for different intensity ranges
            # This ensures even the faintest differences are visible
            
            # Define intensity ranges and corresponding colors (BGR format)
            ranges = [
                (0, 0, (0, 0, 0)),           # Black for zero
                (1, 25, (100, 0, 100)),      # Purple for very faint
                (26, 50, (200, 0, 50)),      # Pink for faint
                (51, 100, (255, 0, 0)),      # Red for moderate
                (101, 150, (255, 100, 0)),   # Orange for strong
                (151, 200, (255, 200, 0)),   # Yellow-orange for very strong
                (201, 255, (255, 255, 0))    # Bright yellow for maximum
            ]
            
            for min_val, max_val, color in ranges:
                mask = (diff_map >= min_val) & (diff_map <= max_val)
                heatmap[mask] = color
                
        elif color_map == 'custom':
            # Custom gradient: Black -> Dark Red -> Red -> Orange -> Yellow -> White
            # Create custom colormap
            colors = [
                (0, 0, 0),      # Black
                (64, 0, 0),     # Dark red
                (128, 0, 0),    # Medium red
                (255, 0, 0),    # Bright red
                (255, 128, 0),  # Orange
                (255, 255, 0),  # Yellow
                (255, 255, 255) # White
            ]
            
            # Create lookup table
            lut = np.zeros((256, 3), dtype=np.uint8)
            n_colors = len(colors)
            
            for i in range(256):
                # Find which color segment we're in
                segment = i * (n_colors - 1) / 255
                idx = int(segment)
                t = segment - idx
                
                if idx >= n_colors - 1:
                    lut[i] = colors[-1]
                else:
                    # Interpolate between colors
                    c1 = np.array(colors[idx])
                    c2 = np.array(colors[idx + 1])
                    lut[i] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
            
            # Apply lookup table (BGR format)
            heatmap = lut[diff_map]
            heatmap = heatmap[:, :, ::-1]  # RGB to BGR
            
        else:
            raise ValueError(f"Unknown color map: {color_map}")
        
        logger.info(f"Created heatmap with {color_map} color mapping")
        
        return heatmap
    
    def create_nonzero_mask_visualization(self) -> np.ndarray:
        """
        Create a visualization showing all non-zero difference areas.
        
        Returns:
            Binary mask visualization as BGR numpy array
        """
        if self.difference_map is None:
            raise ValueError("No difference map calculated")
        
        # Create binary mask
        nonzero_mask = (self.difference_map > 0).astype(np.uint8) * 255
        
        # Create colored visualization
        # Green for non-zero areas, black for zero
        mask_viz = np.zeros((*nonzero_mask.shape, 3), dtype=np.uint8)
        mask_viz[:, :, 1] = nonzero_mask  # Green channel
        
        # Add contours for better visibility
        contours, _ = cv2.findContours(nonzero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_viz, contours, -1, (0, 255, 255), 1)  # Yellow contours
        
        return mask_viz
    
    def overlay_on_original(self, 
                           original_image: np.ndarray,
                           heatmap: np.ndarray,
                           opacity: float = 0.7) -> np.ndarray:
        """
        Overlay the heatmap on the original image.
        
        Args:
            original_image: Original image (grayscale or color)
            heatmap: Heatmap image
            opacity: Opacity of the heatmap overlay (0-1)
            
        Returns:
            Composite image
        """
        # Convert grayscale to BGR if needed
        if len(original_image.shape) == 2:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            original_bgr = original_image.copy()
        
        # Ensure same size
        if original_bgr.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
        
        # Create overlay
        overlay = cv2.addWeighted(original_bgr, 1 - opacity, heatmap, opacity, 0)
        
        return overlay
    
    def analyze_differences(self) -> Dict[str, Any]:
        """
        Analyze the difference map and return statistics.
        
        Returns:
            Dictionary containing analysis results
        """
        if self.difference_map is None:
            raise ValueError("No difference map calculated")
        
        # Find high difference regions
        threshold = np.percentile(self.difference_map, 90)  # Top 10%
        high_diff_mask = self.difference_map > threshold
        
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            high_diff_mask.astype(np.uint8), connectivity=8
        )
        
        # Collect statistics
        analysis = {
            'total_pixels': self.difference_map.size,
            'difference_stats': {
                'min': float(self.difference_map.min()),
                'max': float(self.difference_map.max()),
                'mean': float(self.difference_map.mean()),
                'std': float(self.difference_map.std()),
                'median': float(np.median(self.difference_map))
            },
            'percentiles': {
                f'p{p}': float(np.percentile(self.difference_map, p))
                for p in [10, 25, 50, 75, 90, 95, 99]
            },
            'high_difference_regions': {
                'count': num_labels - 1,  # Exclude background
                'threshold_used': float(threshold),
                'pixels_above_threshold': int(high_diff_mask.sum()),
                'percentage_high_diff': float(high_diff_mask.sum() / self.difference_map.size * 100)
            },
            'regions': []
        }
        
        # Add info about each high-difference region
        for i in range(1, num_labels):  # Skip background (0)
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            
            region_mask = labels == i
            region_values = self.difference_map[region_mask]
            
            analysis['regions'].append({
                'id': i,
                'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'centroid': {'x': float(cx), 'y': float(cy)},
                'area': int(area),
                'max_difference': float(region_values.max()),
                'mean_difference': float(region_values.mean())
            })
        
        # Sort regions by area
        analysis['regions'].sort(key=lambda x: x['area'], reverse=True)
        
        return analysis
    
    def save_outputs(self, 
                    output_dir: str,
                    base_name: str,
                    heatmap: np.ndarray,
                    save_difference_map: bool = True,
                    save_analysis: bool = True,
                    overlay_image: Optional[np.ndarray] = None,
                    save_nonzero_mask: bool = False) -> None:
        """
        Save all output files.
        
        Args:
            output_dir: Output directory
            base_name: Base filename for outputs
            heatmap: Heatmap image to save
            save_difference_map: Whether to save the raw difference map
            save_analysis: Whether to save the analysis JSON
            overlay_image: Optional overlay image to save
            save_nonzero_mask: Whether to save the non-zero mask visualization
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save heatmap
        heatmap_path = output_dir / f"{base_name}_heatmap.png"
        cv2.imwrite(str(heatmap_path), heatmap)
        logger.info(f"Saved heatmap to: {heatmap_path}")
        
        # Save overlay if provided
        if overlay_image is not None:
            overlay_path = output_dir / f"{base_name}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay_image)
            logger.info(f"Saved overlay to: {overlay_path}")
        
        # Save difference map
        if save_difference_map:
            diff_path = output_dir / f"{base_name}_difference_map.npy"
            np.save(diff_path, self.difference_map)
            logger.info(f"Saved difference map to: {diff_path}")
        
        # Save non-zero mask visualization
        if save_nonzero_mask:
            mask_viz = self.create_nonzero_mask_visualization()
            mask_path = output_dir / f"{base_name}_nonzero_mask.png"
            cv2.imwrite(str(mask_path), mask_viz)
            logger.info(f"Saved non-zero mask to: {mask_path}")
        
        # Save analysis
        if save_analysis:
            analysis = self.analyze_differences()
            analysis['source_file'] = str(self.source_file)
            analysis['processing_info'] = {
                'input_shape': self.intensity_matrix.shape,
                'output_shape': heatmap.shape
            }
            
            analysis_path = output_dir / f"{base_name}_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved analysis to: {analysis_path}")
    
    def create_visualization_grid(self, 
                                 heatmap: np.ndarray,
                                 original_image: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive visualization grid.
        
        Args:
            heatmap: Heatmap image
            original_image: Optional original image
            save_path: Optional path to save the visualization
        """
        # Determine number of subplots
        n_plots = 3 if original_image is None else 4
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        if n_plots == 3:
            axes = axes.flatten()
        
        # Plot 1: Intensity matrix
        im1 = axes[0].imshow(self.intensity_matrix, cmap='gray')
        axes[0].set_title('Intensity Matrix')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Plot 2: Difference map
        im2 = axes[1].imshow(self.difference_map, cmap='hot')
        axes[1].set_title('Difference Map')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Plot 3: Heatmap
        # Convert BGR to RGB for display
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        axes[2].imshow(heatmap_rgb)
        axes[2].set_title('Difference Heatmap')
        axes[2].axis('off')
        
        # Plot 4: Original with overlay (if available)
        if original_image is not None:
            overlay = self.overlay_on_original(original_image, heatmap, opacity=0.5)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            axes[3].imshow(overlay_rgb)
            axes[3].set_title('Original with Overlay')
            axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization grid to: {save_path}")
        else:
            plt.show()


def main():
    """Main function to run the intensity difference analyzer."""
    parser = argparse.ArgumentParser(
        description='Generate heatmap images from intensity matrices showing local differences'
    )
    parser.add_argument('input_file', help='Path to intensity matrix file (csv, h5, json, npy)')
    parser.add_argument('-o', '--output', help='Output directory', default='.')
    parser.add_argument('-m', '--method', 
                       choices=['gradient_magnitude', 'max_neighbor', 'sobel', 'laplacian', 'canny_strength'],
                       default='gradient_magnitude',
                       help='Method for calculating differences')
    parser.add_argument('-n', '--neighborhood',
                       choices=['4-connected', '8-connected'],
                       default='8-connected',
                       help='Neighborhood connectivity')
    parser.add_argument('-c', '--colormap',
                       choices=['black_to_red', 'black_red_yellow', 'heat', 'custom', 'highlight'],
                       default='black_to_red',
                       help='Color mapping for heatmap')
    parser.add_argument('-t', '--threshold',
                       type=float, default=0.0,
                       help='Minimum difference threshold (0-255)')
    parser.add_argument('-g', '--gamma',
                       type=float, default=1.0,
                       help='Gamma correction (< 1 emphasizes low values, > 1 emphasizes high values)')
    parser.add_argument('-b', '--blur',
                       type=int, default=0,
                       help='Gaussian blur radius for smoothing')
    parser.add_argument('--highlight-all',
                       action='store_true',
                       help='Highlight all non-zero difference areas with enhanced visibility')
    parser.add_argument('--original',
                       help='Path to original image for overlay')
    parser.add_argument('-v', '--visualize',
                       action='store_true',
                       help='Create visualization grid')
    parser.add_argument('--no-analysis',
                       action='store_true',
                       help='Skip saving analysis JSON')
    parser.add_argument('--no-diff-map',
                       action='store_true',
                       help='Skip saving difference map')
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = IntensityDifferenceAnalyzer()
        
        # Load intensity matrix
        analyzer.load_intensity_matrix(args.input_file)
        
        # Calculate differences
        analyzer.calculate_differences(
            method=args.method,
            neighborhood=args.neighborhood,
            normalize=True
        )
        
        # Create heatmap
        heatmap = analyzer.create_heatmap(
            threshold=args.threshold,
            color_map=args.colormap,
            gamma=args.gamma,
            blur_radius=args.blur if args.blur > 0 else None,
            highlight_all_nonzero=args.highlight_all
        )
        
        # Load original image if provided
        original_image = None
        overlay_image = None
        if args.original:
            original_image = cv2.imread(args.original, cv2.IMREAD_UNCHANGED)
            if original_image is not None:
                overlay_image = analyzer.overlay_on_original(original_image, heatmap)
        
        # Determine base name for outputs
        base_name = Path(args.input_file).stem
        
        # Save outputs
        analyzer.save_outputs(
            output_dir=args.output,
            base_name=base_name,
            heatmap=heatmap,
            save_difference_map=not args.no_diff_map,
            save_analysis=not args.no_analysis,
            overlay_image=overlay_image,
            save_nonzero_mask=args.highlight_all
        )
        
        # Create visualization if requested
        if args.visualize:
            viz_path = Path(args.output) / f"{base_name}_visualization_grid.png"
            analyzer.create_visualization_grid(
                heatmap=heatmap,
                original_image=original_image,
                save_path=viz_path
            )
        
        # Print summary
        analysis = analyzer.analyze_differences()
        print("\n" + "="*50)
        print("DIFFERENCE ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total pixels analyzed: {analysis['total_pixels']:,}")
        print(f"Difference range: [{analysis['difference_stats']['min']:.2f}, {analysis['difference_stats']['max']:.2f}]")
        print(f"Mean difference: {analysis['difference_stats']['mean']:.2f}")
        print(f"High difference regions found: {analysis['high_difference_regions']['count']}")
        print(f"Pixels with high differences: {analysis['high_difference_regions']['percentage_high_diff']:.1f}%")
        
        if analysis['regions'] and len(analysis['regions']) > 0:
            print(f"\nTop 3 largest high-difference regions:")
            for i, region in enumerate(analysis['regions'][:3], 1):
                print(f"  {i}. Area: {region['area']} pixels, "
                      f"Max diff: {region['max_difference']:.1f}, "
                      f"Location: ({region['centroid']['x']:.0f}, {region['centroid']['y']:.0f})")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
