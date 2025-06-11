#!/usr/bin/env python3
"""
Batch process multiple washer images
Usage: python batch_processor.py <directory_or_pattern>
"""

import cv2
import glob
import os
from pathlib import Path
from circle_detector import detect_washer_circles
from washer_splitter import split_washer

def process_batch(pattern, output_dir='batch_output'):
    """Process multiple washer images"""
    # Get file list
    if os.path.isdir(pattern):
        files = glob.glob(os.path.join(pattern, '*.jpg')) + \
                glob.glob(os.path.join(pattern, '*.png')) + \
                glob.glob(os.path.join(pattern, '*.jpeg'))
    else:
        files = glob.glob(pattern)
    
    if not files:
        print(f"No images found matching: {pattern}")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    success = 0
    failed = []
    
    for i, img_path in enumerate(files, 1):
        print(f"\nProcessing {i}/{len(files)}: {os.path.basename(img_path)}")
        
        img = cv2.imread(img_path)
        if img is None:
            failed.append(img_path)
            continue
        
        # Detect and split
        inner, outer = detect_washer_circles(img)
        if inner is None:
            print("  - Failed: No circles detected")
            failed.append(img_path)
            continue
        
        inner_img, ring_img = split_washer(img, inner, outer)
        
        # Save results
        base = Path(img_path).stem
        cv2.imwrite(f'{output_dir}/{base}_inner.png', inner_img)
        cv2.imwrite(f'{output_dir}/{base}_ring.png', ring_img)
        
        print(f"  - Success: r_inner={inner[2]}, r_outer={outer[2]}")
        success += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Processed: {len(files)} images")
    print(f"Success: {success}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFailed images:")
        for f in failed:
            print(f"  - {os.path.basename(f)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python batch_processor.py <directory_or_pattern>")
        print("Example: python batch_processor.py ./washers/")
        print("Example: python batch_processor.py './images/*.jpg'")
    else:
        process_batch(sys.argv[1])