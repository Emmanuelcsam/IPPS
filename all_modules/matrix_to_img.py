#!/usr/bin/env python3
"""
Matrix to Image Converter
========================
Converts intensity matrix data (CSV/JSON) back to images.

Author: Assistant
Date: 2025
"""

import cv2
import numpy as np
import pandas as pd
import json
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MatrixToImageConverter:
    """
    Converts intensity matrices from CSV/JSON back to images.
    """
    
    def __init__(self):
        """Initialize the converter."""
        self.intensity_matrix = None
        self.metadata = {}
        self.source_file = None
        
    def load_from_csv(self, csv_path: str) -> np.ndarray:
        """
        Load intensity matrix from CSV file with coordinates.
        
        Args:
            csv_path: Path to CSV file with x, y, intensity columns
            
        Returns:
            Reconstructed intensity matrix
        """
        logger.info(f"Loading CSV file: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['x', 'y', 'intensity']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Get dimensions
        max_x = int(df['x'].max()) + 1
        max_y = int(df['y'].max()) + 1
        
        logger.info(f"Reconstructing image of size {max_x}x{max_y}")
        
        # Create empty matrix
        self.intensity_matrix = np.zeros((max_y, max_x), dtype=np.float32)
        
        # Fill matrix with intensity values
        for _, row in df.iterrows():
            x = int(row['x'])
            y = int(row['y'])
            intensity = float(row['intensity'])
            self.intensity_matrix[y, x] = intensity
        
        # Store metadata
        self.metadata = {
            'source': 'csv',
            'original_shape': (max_y, max_x),
            'min_intensity': float(self.intensity_matrix.min()),
            'max_intensity': float(self.intensity_matrix.max()),
            'mean_intensity': float(self.intensity_matrix.mean())
        }
        
        logger.info(f"Loaded matrix with shape: {self.intensity_matrix.shape}")
        logger.info(f"Intensity range: [{self.metadata['min_intensity']:.2f}, {self.metadata['max_intensity']:.2f}]")
        
        return self.intensity_matrix
    
    def load_from_json(self, json_path: str) -> np.ndarray:
        """
        Load intensity matrix from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Loaded intensity matrix
        """
        logger.info(f"Loading JSON file: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check for different JSON formats
        if 'intensity_matrix' in data:
            # Standard format from image_to_matrix.py
            self.intensity_matrix = np.array(data['intensity_matrix'], dtype=np.float32)
            if 'metadata' in data:
                self.metadata = data['metadata']
        elif 'defect_matrix' in data:
            # Defect analysis format
            self.intensity_matrix = np.array(data['defect_matrix'], dtype=np.float32)
            self.metadata = data.get('metadata', {})
            self.metadata['source'] = 'defect_analysis'
        else:
            # Try to interpret as raw matrix
            self.intensity_matrix = np.array(data, dtype=np.float32)
            self.metadata = {'source': 'json_raw'}
        
        # Update metadata
        self.metadata.update({
            'original_shape': self.intensity_matrix.shape,
            'min_intensity': float(self.intensity_matrix.min()),
            'max_intensity': float(self.intensity_matrix.max()),
            'mean_intensity': float(self.intensity_matrix.mean())
        })
        
        logger.info(f"Loaded matrix with shape: {self.intensity_matrix.shape}")
        logger.info(f"Intensity range: [{self.metadata['min_intensity']:.2f}, {self.metadata['max_intensity']:.2f}]")
        
        return self.intensity_matrix
    
    def save_as_image(self, output_path: str, normalize: bool = True) -> None:
        """
        Save the intensity matrix as an image file.
        
        Args:
            output_path: Path for output image
            normalize: Whether to normalize to 0-255 range
        """
        if self.intensity_matrix is None:
            raise ValueError("No intensity matrix loaded")
        
        # Prepare image data
        if normalize:
            # Normalize to 0-255 range
            min_val = self.intensity_matrix.min()
            max_val = self.intensity_matrix.max()
            
            if max_val > min_val:
                normalized = (self.intensity_matrix - min_val) / (max_val - min_val) * 255
            else:
                normalized = self.intensity_matrix
            
            image_data = normalized.astype(np.uint8)
        else:
            # Clip to 0-255 range
            image_data = np.clip(self.intensity_matrix, 0, 255).astype(np.uint8)
        
        # Save image
        cv2.imwrite(output_path, image_data)
        logger.info(f"Saved image to: {output_path}")
    
    def create_visualization(self, save_path: Optional[str] = None,
                           colormap: str = 'gray',
                           show_stats: bool = True) -> None:
        """
        Create a visualization of the intensity matrix.
        
        Args:
            save_path: Optional path to save visualization
            colormap: Matplotlib colormap name
            show_stats: Whether to show statistics
        """
        if self.intensity_matrix is None:
            raise ValueError("No intensity matrix loaded")
        
        # Create figure
        if show_stats:
            fig = plt.figure(figsize=(12, 8))
            
            # Main image
            ax1 = plt.subplot(2, 2, (1, 2))
            im = ax1.imshow(self.intensity_matrix, cmap=colormap)
            ax1.set_title('Intensity Matrix Visualization')
            ax1.axis('off')
            plt.colorbar(im, ax=ax1, fraction=0.046)
            
            # Histogram
            ax2 = plt.subplot(2, 2, 3)
            hist_data = self.intensity_matrix.flatten()
            ax2.hist(hist_data, bins=50, color='blue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Intensity Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Intensity Distribution')
            ax2.grid(True, alpha=0.3)
            
            # Statistics text
            ax3 = plt.subplot(2, 2, 4)
            ax3.axis('off')
            
            stats_text = f"""Matrix Statistics:
            
Shape: {self.intensity_matrix.shape}
Min: {self.metadata['min_intensity']:.2f}
Max: {self.metadata['max_intensity']:.2f}
Mean: {self.metadata['mean_intensity']:.2f}
Std: {np.std(self.intensity_matrix):.2f}

Source: {self.metadata.get('source', 'Unknown')}"""
            
            ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle('Intensity Matrix Analysis', fontsize=16)
        else:
            # Simple visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(self.intensity_matrix, cmap=colormap)
            ax.set_title('Intensity Matrix')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to: {save_path}")
        else:
            plt.show()
    
    def apply_colormap(self, colormap_name: str = 'hot') -> np.ndarray:
        """
        Apply a colormap to the intensity matrix.
        
        Args:
            colormap_name: OpenCV colormap name or 'custom'
            
        Returns:
            Colored image (BGR format)
        """
        if self.intensity_matrix is None:
            raise ValueError("No intensity matrix loaded")
        
        # Normalize to 0-255
        normalized = ((self.intensity_matrix - self.intensity_matrix.min()) / 
                     (self.intensity_matrix.max() - self.intensity_matrix.min()) * 255).astype(np.uint8)
        
        # Apply colormap
        colormap_dict = {
            'hot': cv2.COLORMAP_HOT,
            'jet': cv2.COLORMAP_JET,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'hsv': cv2.COLORMAP_HSV,
            'cool': cv2.COLORMAP_COOL,
            'spring': cv2.COLORMAP_SPRING,
            'summer': cv2.COLORMAP_SUMMER,
            'autumn': cv2.COLORMAP_AUTUMN,
            'winter': cv2.COLORMAP_WINTER,
            'bone': cv2.COLORMAP_BONE,
            'ocean': cv2.COLORMAP_OCEAN,
            'parula': cv2.COLORMAP_PARULA,
            'plasma': cv2.COLORMAP_PLASMA,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'inferno': cv2.COLORMAP_INFERNO
        }
        
        if colormap_name in colormap_dict:
            colored = cv2.applyColorMap(normalized, colormap_dict[colormap_name])
        elif colormap_name == 'custom':
            # Custom black to red colormap (good for defects)
            colored = np.zeros((*normalized.shape, 3), dtype=np.uint8)
            colored[:, :, 2] = normalized  # Red channel
        else:
            # Default to grayscale
            colored = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        
        return colored
    
    def convert_defect_matrix(self, defect_colors: Dict[int, Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Convert defect matrix to colored image.
        
        Args:
            defect_colors: Dictionary mapping defect codes to BGR colors
                          Default: {0: black, 1: green, 2: blue, 3: blue}
        
        Returns:
            Colored defect image
        """
        if self.intensity_matrix is None:
            raise ValueError("No matrix loaded")
        
        if defect_colors is None:
            defect_colors = {
                0: (0, 0, 0),       # Background: black
                1: (0, 255, 0),     # Rings: green
                2: (255, 0, 0),     # Scratches: blue
                3: (255, 0, 0)      # Digs: blue
            }
        
        # Create colored image
        height, width = self.intensity_matrix.shape
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply colors based on defect codes
        for code, color in defect_colors.items():
            mask = self.intensity_matrix == code
            colored_image[mask] = color
        
        return colored_image


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert intensity matrix data back to images"
    )
    parser.add_argument('input_file', help='Input CSV or JSON file')
    parser.add_argument('-o', '--output', help='Output image file (default: input_name.png)')
    parser.add_argument('-c', '--colormap', default='gray',
                       choices=['gray', 'hot', 'jet', 'rainbow', 'hsv', 'cool', 
                               'spring', 'summer', 'autumn', 'winter', 'bone',
                               'ocean', 'parula', 'plasma', 'viridis', 'inferno', 'custom'],
                       help='Colormap to apply (default: gray)')
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='Create visualization with statistics')
    parser.add_argument('--viz-output', help='Path for visualization output')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Do not normalize intensity values')
    parser.add_argument('--defect-mode', action='store_true',
                       help='Treat as defect matrix (0=bg, 1=rings, 2=scratches, 3=digs)')
    
    args = parser.parse_args()
    
    # Determine input type
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Create converter
    converter = MatrixToImageConverter()
    
    try:
        # Load data
        if input_path.suffix.lower() == '.csv':
            converter.load_from_csv(str(input_path))
        elif input_path.suffix.lower() == '.json':
            converter.load_from_json(str(input_path))
        else:
            print(f"Error: Unsupported file type: {input_path.suffix}")
            sys.exit(1)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = str(input_path.with_suffix('.png'))
        
        # Save image
        if args.defect_mode:
            # Convert defect matrix to colored image
            colored_image = converter.convert_defect_matrix()
            cv2.imwrite(output_path, colored_image)
            logger.info(f"Saved defect visualization to: {output_path}")
        else:
            # Apply colormap if requested
            if args.colormap != 'gray':
                colored_image = converter.apply_colormap(args.colormap)
                cv2.imwrite(output_path, colored_image)
                logger.info(f"Saved colored image to: {output_path}")
            else:
                # Save as grayscale
                converter.save_as_image(output_path, normalize=not args.no_normalize)
        
        # Create visualization if requested
        if args.visualize:
            if args.viz_output:
                viz_path = args.viz_output
            else:
                # Properly construct visualization filename
                viz_path = str(input_path.parent / (input_path.stem + '_viz.png'))
            converter.create_visualization(viz_path, colormap=args.colormap)
        
        print(f"\nConversion complete!")
        print(f"Output image: {output_path}")
        if args.visualize:
            if args.viz_output:
                print(f"Visualization: {args.viz_output}")
            else:
                print(f"Visualization: {input_path.parent / (input_path.stem + '_viz.png')}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
