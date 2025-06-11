#!/usr/bin/env python3
"""
Command-line tool for washer image processing
Usage: python cli_tool.py <image_path> [--display] [--save] [--output-dir <dir>]
"""

import cv2
import argparse
from pathlib import Path
from circle_detector import detect_washer_circles
from washer_splitter import split_washer
from visualizer import display_results, draw_circles

def process_image(img_path, display=False, save=False, output_dir='output'):
    """Process a single washer image"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot read {img_path}")
        return False
    
    # Detect circles
    inner, outer = detect_washer_circles(img)
    if inner is None:
        print(f"Failed to detect circles in {img_path}")
        return False
    
    # Split image
    inner_img, ring_img = split_washer(img, inner, outer)
    
    # Display if requested
    if display:
        display_results(img, inner_img, ring_img, inner, outer)
    
    # Save if requested
    if save:
        Path(output_dir).mkdir(exist_ok=True)
        base = Path(img_path).stem
        
        cv2.imwrite(f'{output_dir}/{base}_inner.png', inner_img)
        cv2.imwrite(f'{output_dir}/{base}_ring.png', ring_img)
        
        vis = draw_circles(img, inner, outer)
        cv2.imwrite(f'{output_dir}/{base}_vis.png', vis)
        
        print(f"Saved: {base}_inner.png, {base}_ring.png, {base}_vis.png")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Split washer images')
    parser.add_argument('image', help='Path to washer image')
    parser.add_argument('-d', '--display', action='store_true', help='Display results')
    parser.add_argument('-s', '--save', action='store_true', help='Save results')
    parser.add_argument('-o', '--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Default to save if nothing specified
    if not args.display and not args.save:
        args.save = True
    
    process_image(args.image, args.display, args.save, args.output_dir)

if __name__ == "__main__":
    main()