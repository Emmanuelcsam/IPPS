#!/usr/bin/env python3
"""
Image to Pixel Intensity Matrix Converter
=========================================
This script converts an image to a matrix database where each position (x,y)
contains the pixel intensity value at that location.

Author: Assistant
Date: 2025
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import h5py
import argparse
import sys
from typing import Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageToMatrixConverter:
    """
    A class to convert images to pixel intensity matrices with various export options.
    """
    
    def __init__(self, image_path: str):
        """
        Initialize the converter with an image path.
        
        Args:
            image_path: Path to the input image
        """
        self.image_path = Path(image_path)
        self.original_image = None
        self.grayscale_image = None
        self.intensity_matrix = None
        self.image_info = {}
        
        # Validate image path
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load the image
        self._load_image()
    
    def _load_image(self) -> None:
        """Load the image and store basic information."""
        # Read image in original format
        self.original_image = cv2.imread(str(self.image_path), cv2.IMREAD_UNCHANGED)
        
        if self.original_image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        
        # Store image information
        self.image_info = {
            'filename': self.image_path.name,
            'original_shape': self.original_image.shape,
            'dtype': str(self.original_image.dtype),
            'channels': 1 if len(self.original_image.shape) == 2 else self.original_image.shape[2]
        }
        
        logger.info(f"Loaded image: {self.image_info['filename']}")
        logger.info(f"Original shape: {self.image_info['original_shape']}")
        logger.info(f"Data type: {self.image_info['dtype']}")
        logger.info(f"Channels: {self.image_info['channels']}")
    
    def convert_to_intensity_matrix(self, method: str = 'luminance') -> np.ndarray:
        """
        Convert the image to a pixel intensity matrix.
        
        Args:
            method: Method for calculating intensity
                   - 'luminance': Standard luminance formula (default)
                   - 'average': Simple average of RGB channels
                   - 'max': Maximum value across channels
                   - 'min': Minimum value across channels
        
        Returns:
            Numpy array containing pixel intensities
        """
        if len(self.original_image.shape) == 2:
            # Already grayscale
            self.grayscale_image = self.original_image.copy()
            logger.info("Image is already grayscale")
        else:
            # Convert color image to grayscale
            if method == 'luminance':
                # Standard luminance formula: 0.299*R + 0.587*G + 0.114*B
                if self.original_image.shape[2] >= 3:
                    # OpenCV uses BGR format
                    b, g, r = self.original_image[:, :, 0], self.original_image[:, :, 1], self.original_image[:, :, 2]
                    self.grayscale_image = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
                else:
                    # If less than 3 channels, use average
                    self.grayscale_image = np.mean(self.original_image, axis=2).astype(np.uint8)
            elif method == 'average':
                self.grayscale_image = np.mean(self.original_image, axis=2).astype(np.uint8)
            elif method == 'max':
                self.grayscale_image = np.max(self.original_image, axis=2).astype(np.uint8)
            elif method == 'min':
                self.grayscale_image = np.min(self.original_image, axis=2).astype(np.uint8)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            logger.info(f"Converted to grayscale using {method} method")
        
        # Create intensity matrix (ensure proper data type)
        self.intensity_matrix = self.grayscale_image.astype(np.float32)
        
        # Store additional information
        self.image_info['intensity_method'] = method
        self.image_info['intensity_shape'] = self.intensity_matrix.shape
        self.image_info['intensity_min'] = float(np.min(self.intensity_matrix))
        self.image_info['intensity_max'] = float(np.max(self.intensity_matrix))
        self.image_info['intensity_mean'] = float(np.mean(self.intensity_matrix))
        self.image_info['intensity_std'] = float(np.std(self.intensity_matrix))
        
        logger.info(f"Intensity matrix shape: {self.intensity_matrix.shape}")
        logger.info(f"Intensity range: [{self.image_info['intensity_min']:.2f}, {self.image_info['intensity_max']:.2f}]")
        logger.info(f"Intensity mean: {self.image_info['intensity_mean']:.2f}")
        logger.info(f"Intensity std: {self.image_info['intensity_std']:.2f}")
        
        return self.intensity_matrix
    
    def get_pixel_intensity(self, x: int, y: int) -> float:
        """
        Get the intensity value at a specific pixel location.
        
        Args:
            x: X coordinate (column)
            y: Y coordinate (row)
        
        Returns:
            Intensity value at (x, y)
        """
        if self.intensity_matrix is None:
            raise ValueError("Intensity matrix not yet created. Run convert_to_intensity_matrix() first.")
        
        height, width = self.intensity_matrix.shape
        if x < 0 or x >= width or y < 0 or y >= height:
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds. Image size: {width}x{height}")
        
        return self.intensity_matrix[y, x]
    
    def save_as_csv(self, output_path: str, include_coordinates: bool = True) -> None:
        """
        Save the intensity matrix as a CSV file.
        
        Args:
            output_path: Path for the output CSV file
            include_coordinates: If True, include x,y coordinates in the CSV
        """
        if self.intensity_matrix is None:
            raise ValueError("Intensity matrix not yet created. Run convert_to_intensity_matrix() first.")
        
        if include_coordinates:
            # Create a dataframe with x, y, and intensity columns
            height, width = self.intensity_matrix.shape
            data = []
            
            for y in range(height):
                for x in range(width):
                    data.append({
                        'x': x,
                        'y': y,
                        'intensity': self.intensity_matrix[y, x]
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved CSV with coordinates to: {output_path}")
        else:
            # Save as a 2D matrix
            np.savetxt(output_path, self.intensity_matrix, delimiter=',', fmt='%.2f')
            logger.info(f"Saved CSV matrix to: {output_path}")
    
    def save_as_numpy(self, output_path: str) -> None:
        """
        Save the intensity matrix as a NumPy binary file.
        
        Args:
            output_path: Path for the output .npy file
        """
        if self.intensity_matrix is None:
            raise ValueError("Intensity matrix not yet created. Run convert_to_intensity_matrix() first.")
        
        np.save(output_path, self.intensity_matrix)
        logger.info(f"Saved NumPy array to: {output_path}")
    
    def save_as_hdf5(self, output_path: str) -> None:
        """
        Save the intensity matrix as an HDF5 file with metadata.
        
        Args:
            output_path: Path for the output .h5 file
        """
        if self.intensity_matrix is None:
            raise ValueError("Intensity matrix not yet created. Run convert_to_intensity_matrix() first.")
        
        with h5py.File(output_path, 'w') as f:
            # Save the intensity matrix
            dset = f.create_dataset('intensity_matrix', data=self.intensity_matrix, compression='gzip')
            
            # Save metadata
            for key, value in self.image_info.items():
                dset.attrs[key] = value
        
        logger.info(f"Saved HDF5 file to: {output_path}")
    
    def save_as_json(self, output_path: str, compact: bool = False) -> None:
        """
        Save the intensity matrix as a JSON file.
        
        Args:
            output_path: Path for the output JSON file
            compact: If True, save in compact format
        """
        if self.intensity_matrix is None:
            raise ValueError("Intensity matrix not yet created. Run convert_to_intensity_matrix() first.")
        
        data = {
            'metadata': self.image_info,
            'intensity_matrix': self.intensity_matrix.tolist()
        }
        
        with open(output_path, 'w') as f:
            if compact:
                json.dump(data, f, separators=(',', ':'))
            else:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved JSON file to: {output_path}")
    
    def visualize_comparison(self, output_path: Optional[str] = None) -> None:
        """
        Create a visualization comparing the original image and intensity matrix.
        
        Args:
            output_path: Optional path to save the visualization
        """
        if self.intensity_matrix is None:
            raise ValueError("Intensity matrix not yet created. Run convert_to_intensity_matrix() first.")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(self.original_image.shape) == 3:
            # Convert BGR to RGB for display
            rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            axes[0].imshow(rgb_image)
        else:
            axes[0].imshow(self.original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Intensity matrix as grayscale
        im1 = axes[1].imshow(self.intensity_matrix, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title('Intensity Matrix (Grayscale)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Intensity matrix as heatmap
        im2 = axes[2].imshow(self.intensity_matrix, cmap='hot')
        axes[2].set_title('Intensity Matrix (Heatmap)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to: {output_path}")
        else:
            plt.show()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the intensity matrix.
        
        Returns:
            Dictionary containing various statistics
        """
        if self.intensity_matrix is None:
            raise ValueError("Intensity matrix not yet created. Run convert_to_intensity_matrix() first.")
        
        stats = {
            'basic_info': self.image_info,
            'intensity_distribution': {
                'min': float(np.min(self.intensity_matrix)),
                'max': float(np.max(self.intensity_matrix)),
                'mean': float(np.mean(self.intensity_matrix)),
                'median': float(np.median(self.intensity_matrix)),
                'std': float(np.std(self.intensity_matrix)),
                'var': float(np.var(self.intensity_matrix))
            },
            'percentiles': {
                f'p{p}': float(np.percentile(self.intensity_matrix, p))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            },
            'pixel_count': {
                'total': int(self.intensity_matrix.size),
                'unique_values': int(np.unique(self.intensity_matrix).size)
            }
        }
        
        return stats


def main():
    """Main function to run the image to matrix converter."""
    parser = argparse.ArgumentParser(
        description='Convert an image to a pixel intensity matrix database'
    )
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('-o', '--output', help='Output directory (default: current directory)', 
                       default='.')
    parser.add_argument('-m', '--method', 
                       choices=['luminance', 'average', 'max', 'min'],
                       default='luminance',
                       help='Method for calculating intensity (default: luminance)')
    parser.add_argument('-f', '--formats', 
                       nargs='+',
                       choices=['csv', 'csv_coords', 'numpy', 'hdf5', 'json'],
                       default=['numpy'],
                       help='Output formats (default: numpy)')
    parser.add_argument('-v', '--visualize', 
                       action='store_true',
                       help='Create visualization comparison')
    parser.add_argument('-s', '--stats', 
                       action='store_true',
                       help='Print detailed statistics')
    
    args = parser.parse_args()
    
    try:
        # Create converter instance
        converter = ImageToMatrixConverter(args.image_path)
        
        # Convert to intensity matrix
        converter.convert_to_intensity_matrix(method=args.method)
        
        # Create output directory if needed
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Base filename for outputs
        base_name = converter.image_path.stem
        
        # Save in requested formats
        for fmt in args.formats:
            if fmt == 'csv':
                output_path = output_dir / f"{base_name}_intensity_matrix.csv"
                converter.save_as_csv(output_path, include_coordinates=False)
            elif fmt == 'csv_coords':
                output_path = output_dir / f"{base_name}_intensity_with_coords.csv"
                converter.save_as_csv(output_path, include_coordinates=True)
            elif fmt == 'numpy':
                output_path = output_dir / f"{base_name}_intensity_matrix.npy"
                converter.save_as_numpy(output_path)
            elif fmt == 'hdf5':
                output_path = output_dir / f"{base_name}_intensity_matrix.h5"
                converter.save_as_hdf5(output_path)
            elif fmt == 'json':
                output_path = output_dir / f"{base_name}_intensity_matrix.json"
                converter.save_as_json(output_path)
        
        # Create visualization if requested
        if args.visualize:
            viz_path = output_dir / f"{base_name}_visualization.png"
            converter.visualize_comparison(viz_path)
        
        # Print statistics if requested
        if args.stats:
            stats = converter.get_statistics()
            print("\n" + "="*50)
            print("INTENSITY MATRIX STATISTICS")
            print("="*50)
            print(json.dumps(stats, indent=2))
        
        # Example: Access specific pixel
        h, w = converter.intensity_matrix.shape
        center_x, center_y = w // 2, h // 2
        center_intensity = converter.get_pixel_intensity(center_x, center_y)
        logger.info(f"\nExample - Center pixel ({center_x}, {center_y}) intensity: {center_intensity:.2f}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
