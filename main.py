#!/usr/bin/env python3
"""
Fiber Optic End Face Inspection System
=====================================
Main script that integrates image processing, difference analysis,
and defect detection into a unified inspection workflow.

"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import shutil
from datetime import datetime
import cv2
import numpy as np

# Import the modules directly if they're in the same directory
try:
    from image_to_matrix import ImageToMatrixConverter
    from heatmap import IntensityDifferenceAnalyzer
    from pixel_defects import FiberDefectDetector
    MODULES_IMPORTED = True
except ImportError:
    MODULES_IMPORTED = False
    print("Warning: Could not import modules directly. Will use subprocess calls.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FiberInspectionSystem:
    """
    Unified system for fiber optic end face inspection.
    """
    
    def __init__(self):
        """Initialize the inspection system."""
        self.config = {}
        self.results = {}
        self.output_dir = None
        self.session_id = None
        
    def display_banner(self):
        """Display the system banner."""
        banner = """
╔═══════════════════════════════════════════════════════════════════╗
║           FIBER OPTIC END FACE INSPECTION SYSTEM                  ║
║                                                                   ║
║  Automated Detection of Scratches and Digs in Fiber Optics       ║
║                         Version 1.0                               ║
╚═══════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def get_user_input(self, prompt: str, default: Any = None, 
                      input_type: type = str, choices: List = None,
                      validate_func: callable = None) -> Any:
        """
        Get validated user input with optional default value.
        
        Args:
            prompt: Input prompt
            default: Default value if user presses enter
            input_type: Expected type of input
            choices: List of valid choices
            validate_func: Custom validation function
            
        Returns:
            Validated user input
        """
        while True:
            if default is not None:
                full_prompt = f"{prompt} [{default}]: "
            else:
                full_prompt = f"{prompt}: "
            
            if choices:
                print(f"Options: {', '.join(map(str, choices))}")
            
            user_input = input(full_prompt).strip()
            
            # Use default if empty
            if not user_input and default is not None:
                return default
            
            # Validate input
            try:
                # Type conversion
                if input_type == bool:
                    value = user_input.lower() in ['yes', 'y', 'true', '1']
                else:
                    value = input_type(user_input)
                
                # Check choices
                if choices and value not in choices:
                    print(f"Error: Please choose from {choices}")
                    continue
                
                # Custom validation
                if validate_func and not validate_func(value):
                    print("Error: Invalid input")
                    continue
                
                return value
                
            except ValueError:
                print(f"Error: Expected {input_type.__name__} input")
                continue
    
    def configure_scan(self):
        """Interactive configuration for the scan."""
        print("\n" + "="*60)
        print("SCAN CONFIGURATION")
        print("="*60)
        
        # Get input image
        while True:
            image_path = self.get_user_input(
                "Enter the path to the fiber end face image",
                input_type=str
            )
            if Path(image_path).exists():
                self.config['image_path'] = image_path
                break
            print("Error: File not found. Please enter a valid path.")
        
        # Quick or detailed configuration
        quick_mode = self.get_user_input(
            "Use quick scan with default settings? (yes/no)",
            default="yes",
            input_type=bool
        )
        
        if quick_mode:
            # Use optimized defaults
            self.config.update({
                'intensity_method': 'luminance',
                'output_formats': ['numpy', 'json'],
                'difference_method': 'gradient_magnitude',
                'neighborhood': '8-connected',
                'colormap': 'black_to_red',
                'highlight_all': True,
                'threshold': 0.0,
                'gamma': 0.5,
                'blur': 0,
                'num_rings': 2,
                'min_scratch_length': 20,
                'min_dig_area': 10,
                'enhancement_factor': 2.0,
                'create_visualizations': True
            })
            print("\nUsing optimized default settings for fiber inspection.")
        else:
            # Detailed configuration
            self._configure_intensity_conversion()
            self._configure_difference_analysis()
            self._configure_defect_detection()
        
        # Output directory
        default_output = f"fiber_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = self.get_user_input(
            "Output directory name",
            default=default_output,
            input_type=str
        )
        
        # Create output directory
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = self.output_dir / "scan_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"\nConfiguration saved to: {config_path}")
    
    def _configure_intensity_conversion(self):
        """Configure intensity matrix conversion settings."""
        print("\n--- Intensity Matrix Conversion ---")
        
        self.config['intensity_method'] = self.get_user_input(
            "Intensity calculation method",
            default="luminance",
            choices=['luminance', 'average', 'max', 'min']
        )
        
        formats = []
        if self.get_user_input("Save as NumPy array?", default="yes", input_type=bool):
            formats.append('numpy')
        if self.get_user_input("Save as JSON?", default="yes", input_type=bool):
            formats.append('json')
        if self.get_user_input("Save as CSV?", default="no", input_type=bool):
            formats.append('csv_coords')
        if self.get_user_input("Save as HDF5?", default="no", input_type=bool):
            formats.append('hdf5')
        
        self.config['output_formats'] = formats
    
    def _configure_difference_analysis(self):
        """Configure difference analysis settings."""
        print("\n--- Difference Analysis ---")
        
        self.config['difference_method'] = self.get_user_input(
            "Difference calculation method",
            default="gradient_magnitude",
            choices=['gradient_magnitude', 'max_neighbor', 'sobel', 'laplacian', 'canny_strength']
        )
        
        self.config['neighborhood'] = self.get_user_input(
            "Neighborhood connectivity",
            default="8-connected",
            choices=['4-connected', '8-connected']
        )
        
        self.config['colormap'] = self.get_user_input(
            "Heatmap color scheme",
            default="black_to_red",
            choices=['black_to_red', 'black_red_yellow', 'heat', 'custom', 'highlight']
        )
        
        self.config['highlight_all'] = self.get_user_input(
            "Highlight all non-zero differences?",
            default="yes",
            input_type=bool
        )
        
        self.config['threshold'] = self.get_user_input(
            "Minimum difference threshold (0-255)",
            default=0.0,
            input_type=float,
            validate_func=lambda x: 0 <= x <= 255
        )
        
        self.config['gamma'] = self.get_user_input(
            "Gamma correction (< 1 enhances faint features)",
            default=0.5,
            input_type=float,
            validate_func=lambda x: 0.1 <= x <= 3.0
        )
        
        self.config['blur'] = self.get_user_input(
            "Gaussian blur radius (0 for none)",
            default=0,
            input_type=int,
            validate_func=lambda x: 0 <= x <= 10
        )
    
    def _configure_defect_detection(self):
        """Configure defect detection settings."""
        print("\n--- Defect Detection ---")
        
        self.config['num_rings'] = self.get_user_input(
            "Expected number of fiber rings",
            default=2,
            input_type=int,
            validate_func=lambda x: 0 <= x <= 5
        )
        
        self.config['min_scratch_length'] = self.get_user_input(
            "Minimum scratch length (pixels)",
            default=20,
            input_type=int,
            validate_func=lambda x: x > 0
        )
        
        self.config['min_dig_area'] = self.get_user_input(
            "Minimum dig area (pixels)",
            default=10,
            input_type=int,
            validate_func=lambda x: x > 0
        )
        
        self.config['enhancement_factor'] = self.get_user_input(
            "Enhancement factor for faint defects",
            default=2.0,
            input_type=float,
            validate_func=lambda x: 0.5 <= x <= 5.0
        )
        
        self.config['create_visualizations'] = self.get_user_input(
            "Create visualization reports?",
            default="yes",
            input_type=bool
        )
    
    def run_integrated_pipeline(self):
        """Run the integrated inspection pipeline using imported modules."""
        print("\n" + "="*60)
        print("RUNNING INTEGRATED INSPECTION PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Convert image to intensity matrix
            print("\n[1/3] Converting image to intensity matrix...")
            converter = ImageToMatrixConverter(self.config['image_path'])
            intensity_matrix = converter.convert_to_intensity_matrix(
                method=self.config['intensity_method']
            )
            
            # Save outputs
            base_name = Path(self.config['image_path']).stem
            for fmt in self.config['output_formats']:
                if fmt == 'numpy':
                    output_path = self.output_dir / f"{base_name}_intensity_matrix.npy"
                    converter.save_as_numpy(str(output_path))
                elif fmt == 'json':
                    output_path = self.output_dir / f"{base_name}_intensity_matrix.json"
                    converter.save_as_json(str(output_path))
                elif fmt == 'csv_coords':
                    output_path = self.output_dir / f"{base_name}_intensity_with_coords.csv"
                    converter.save_as_csv(str(output_path), include_coordinates=True)
                elif fmt == 'hdf5':
                    output_path = self.output_dir / f"{base_name}_intensity_matrix.h5"
                    converter.save_as_hdf5(str(output_path))
            
            if self.config['create_visualizations']:
                viz_path = self.output_dir / f"{base_name}_visualization.png"
                converter.visualize_comparison(str(viz_path))
            
            # Store results
            self.results['intensity_stats'] = converter.get_statistics()
            
            # Step 2: Analyze differences
            print("\n[2/3] Analyzing pixel intensity differences...")
            analyzer = IntensityDifferenceAnalyzer()
            analyzer.intensity_matrix = intensity_matrix
            
            difference_map = analyzer.calculate_differences(
                method=self.config['difference_method'],
                neighborhood=self.config['neighborhood'],
                normalize=True
            )
            
            heatmap = analyzer.create_heatmap(
                threshold=self.config['threshold'],
                color_map=self.config['colormap'],
                gamma=self.config['gamma'],
                blur_radius=self.config['blur'] if self.config['blur'] > 0 else None,
                highlight_all_nonzero=self.config['highlight_all']
            )
            
            # Save heatmap outputs
            analyzer.save_outputs(
                output_dir=str(self.output_dir),
                base_name=f"{base_name}_intensity_matrix",
                heatmap=heatmap,
                save_difference_map=True,
                save_analysis=True,
                save_nonzero_mask=self.config['highlight_all']
            )
            
            if self.config['create_visualizations']:
                viz_path = self.output_dir / f"{base_name}_intensity_matrix_visualization_grid.png"
                analyzer.create_visualization_grid(
                    heatmap=heatmap,
                    original_image=converter.original_image,
                    save_path=str(viz_path)
                )
            
            # Store results
            self.results['difference_analysis'] = analyzer.analyze_differences()
            
            # Step 3: Detect defects
            print("\n[3/3] Detecting scratches and digs...")
            detector = FiberDefectDetector()
            detector.processed_map = difference_map
            detector.intensity_matrix = intensity_matrix
            
            # Detect rings
            if self.config['num_rings'] > 0:
                detector.detect_fiber_rings(num_rings=self.config['num_rings'])
                detector.create_ring_mask()
            
            # Enhance and detect
            detector.enhance_faint_defects(
                enhancement_factor=self.config['enhancement_factor']
            )
            
            scratches = detector.detect_scratches(
                min_length=self.config['min_scratch_length']
            )
            digs = detector.detect_digs(
                min_area=self.config['min_dig_area']
            )
            
            # Save results
            detector.save_results(
                output_dir=str(self.output_dir),
                base_name=f"{base_name}_intensity_matrix"
            )
            
            # Store results
            self.results['defect_detection'] = {
                'rings_detected': len(detector.detected_rings),
                'scratches_detected': len(scratches),
                'digs_detected': len(digs),
                'total_defects': len(scratches) + len(digs)
            }
            
            print("\n✓ Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_subprocess_pipeline(self):
        """Run the pipeline using subprocess calls to individual scripts."""
        print("\n" + "="*60)
        print("RUNNING INSPECTION PIPELINE (Subprocess Mode)")
        print("="*60)
        
        base_name = Path(self.config['image_path']).stem
        
        # Step 1: Image to Matrix
        print("\n[1/3] Converting image to intensity matrix...")
        cmd1 = [
            sys.executable, "image_to_matrix.py",
            self.config['image_path'],
            "-o", str(self.output_dir),
            "-m", self.config['intensity_method'],
            "-f"
        ] + self.config['output_formats']
        
        if self.config['create_visualizations']:
            cmd1.append("-v")
        
        result = subprocess.run(cmd1, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in step 1: {result.stderr}")
            return False
        print(result.stdout)
        
        # Step 2: Heatmap Generation
        print("\n[2/3] Analyzing pixel intensity differences...")
        intensity_file = self.output_dir / f"{base_name}_intensity_matrix.npy"
        
        cmd2 = [
            sys.executable, "heatmap.py",
            str(intensity_file),
            "-o", str(self.output_dir),
            "-m", self.config['difference_method'],
            "-n", self.config['neighborhood'],
            "-c", self.config['colormap'],
            "-t", str(self.config['threshold']),
            "-g", str(self.config['gamma']),
            "-b", str(self.config['blur'])
        ]
        
        if self.config['highlight_all']:
            cmd2.append("--highlight-all")
        if self.config['create_visualizations']:
            cmd2.append("-v")
            cmd2.extend(["--original", self.config['image_path']])
        
        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in step 2: {result.stderr}")
            return False
        print(result.stdout)
        
        # Step 3: Defect Detection
        print("\n[3/3] Detecting scratches and digs...")
        json_file = self.output_dir / f"{base_name}_intensity_matrix_analysis.json"
        heatmap_file = self.output_dir / f"{base_name}_intensity_matrix_heatmap.png"
        
        cmd3 = [
            sys.executable, "pixel_defects.py",
            "--json", str(json_file),
            "--image", str(heatmap_file),
            "-o", str(self.output_dir),
            "--rings", str(self.config['num_rings']),
            "--min-scratch-length", str(self.config['min_scratch_length']),
            "--min-dig-area", str(self.config['min_dig_area']),
            "--enhancement", str(self.config['enhancement_factor'])
        ]
        
        result = subprocess.run(cmd3, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in step 3: {result.stderr}")
            return False
        print(result.stdout)
        
        print("\n✓ Pipeline completed successfully!")
        return True
    
    def generate_report(self):
        """Generate a comprehensive inspection report."""
        print("\n" + "="*60)
        print("INSPECTION REPORT")
        print("="*60)
        
        report_path = self.output_dir / "inspection_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("FIBER OPTIC END FACE INSPECTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image: {self.config['image_path']}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("CONFIGURATION SUMMARY\n")
            f.write("-"*30 + "\n")
            for key, value in self.config.items():
                if key != 'image_path':
                    f.write(f"{key}: {value}\n")
            
            if self.results:
                f.write("\n\nRESULTS SUMMARY\n")
                f.write("-"*30 + "\n")
                
                if 'intensity_stats' in self.results:
                    stats = self.results['intensity_stats']['intensity_distribution']
                    f.write(f"\nIntensity Statistics:\n")
                    f.write(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n")
                    f.write(f"  Mean: {stats['mean']:.2f}\n")
                    f.write(f"  Std Dev: {stats['std']:.2f}\n")
                
                if 'defect_detection' in self.results:
                    defects = self.results['defect_detection']
                    f.write(f"\nDefect Detection Results:\n")
                    f.write(f"  Fiber Rings Detected: {defects['rings_detected']}\n")
                    f.write(f"  Scratches Found: {defects['scratches_detected']}\n")
                    f.write(f"  Digs Found: {defects['digs_detected']}\n")
                    f.write(f"  Total Defects: {defects['total_defects']}\n")
        
        print(f"\nReport saved to: {report_path}")
        
        # List all generated files
        print("\nGenerated Files:")
        for file in sorted(self.output_dir.glob("*")):
            if file.is_file():
                print(f"  - {file.name}")
    
    def run(self):
        """Run the complete fiber inspection system."""
        self.display_banner()
        
        try:
            # Configure the scan
            self.configure_scan()
            
            # Run the pipeline
            if MODULES_IMPORTED:
                self.run_integrated_pipeline()
            else:
                success = self.run_subprocess_pipeline()
                if not success:
                    print("\nPipeline failed. Check error messages above.")
                    return
            
            # Generate report
            self.generate_report()
            
            print("\n" + "="*60)
            print("INSPECTION COMPLETE")
            print("="*60)
            print(f"All results saved to: {self.output_dir}")
            
        except KeyboardInterrupt:
            print("\n\nInspection cancelled by user.")
        except Exception as e:
            print(f"\n\nError during inspection: {e}")
            logger.error("Inspection failed", exc_info=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fiber Optic End Face Inspection System"
    )
    parser.add_argument(
        '--batch',
        help='Run in batch mode with a configuration file',
        type=str
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick scan with optimal defaults'
    )
    
    args = parser.parse_args()
    
    system = FiberInspectionSystem()
    
    if args.batch:
        # Load configuration from file
        with open(args.batch, 'r') as f:
            system.config = json.load(f)
        system.output_dir = Path(system.config.get('output_dir', 'fiber_inspection_batch'))
        system.output_dir.mkdir(parents=True, exist_ok=True)
        
        if MODULES_IMPORTED:
            system.run_integrated_pipeline()
        else:
            system.run_subprocess_pipeline()
        
        system.generate_report()
    else:
        # Interactive mode
        system.run()


if __name__ == "__main__":
    main()
