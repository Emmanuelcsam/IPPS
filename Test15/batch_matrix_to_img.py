#!/usr/bin/env python3
"""
Convert All Fiber Inspection Outputs to Images
=============================================
Batch converts all CSV and JSON output files to images.

Author: Assistant
Date: 2025
"""

import os
import sys
import glob
from pathlib import Path
import subprocess

def convert_inspection_outputs(directory: str):
    """
    Convert all inspection output files in a directory to images.
    
    Args:
        directory: Directory containing inspection output files
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Error: Directory not found: {directory}")
        return
    
    print(f"Converting files in: {directory}")
    print("="*60)
    
    # Convert intensity matrix CSV files
    csv_files = list(dir_path.glob("*_intensity_with_coords.csv"))
    for csv_file in csv_files:
        print(f"\nConverting: {csv_file.name}")
        output_name = csv_file.stem + "_reconstructed.png"
        output_path = dir_path / output_name
        
        cmd = [
            sys.executable, "matrix_to_img.py",
            str(csv_file),
            "-o", str(output_path),
            "-c", "hot",
            "-v",
            "--viz-output", str(dir_path / (csv_file.stem + "_analysis.png"))
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Created: {output_name}")
        else:
            print(f"✗ Failed: {result.stderr}")
    
    # Convert intensity matrix JSON files
    json_files = list(dir_path.glob("*_intensity_matrix.json"))
    for json_file in json_files:
        if "defect" not in json_file.name:  # Skip defect analysis files
            print(f"\nConverting: {json_file.name}")
            output_name = json_file.stem + "_from_json.png"
            output_path = dir_path / output_name
            
            cmd = [
                sys.executable, "matrix_to_img.py",
                str(json_file),
                "-o", str(output_path),
                "-c", "viridis"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Created: {output_name}")
            else:
                print(f"✗ Failed: {result.stderr}")
    
    # Convert defect analysis JSON files
    defect_files = list(dir_path.glob("*_defect_analysis.json"))
    for defect_file in defect_files:
        print(f"\nConverting: {defect_file.name}")
        output_name = defect_file.stem + "_visual.png"
        output_path = dir_path / output_name
        
        cmd = [
            sys.executable, "matrix_to_img.py",
            str(defect_file),
            "-o", str(output_path),
            "--defect-mode"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Created: {output_name}")
        else:
            print(f"✗ Failed: {result.stderr}")
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"Check {directory} for generated images")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert all fiber inspection outputs to images"
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Directory containing output files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    convert_inspection_outputs(args.directory)


if __name__ == "__main__":
    main()
