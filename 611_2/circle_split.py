import cv2
import numpy as np
import os
from pathlib import Path

class WasherSplitter:
    def __init__(self, image_path):
        """Initialize with image path"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        
    def detect_circles_hough(self):
        """Detect circles using Hough Transform"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=min(self.width, self.height) // 2
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0]
        return None
    
    def detect_circles_contours(self):
        """Detect circles using contour detection"""
        # Apply threshold
        _, thresh = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                # Fit circle to contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Check if contour is circular enough
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.7:  # Threshold for circular shape
                    valid_contours.append((center[0], center[1], radius, area))
        
        # Sort by area (largest first)
        valid_contours.sort(key=lambda x: x[3], reverse=True)
        
        return valid_contours[:2] if len(valid_contours) >= 2 else None
    
    def split_washer(self, method='auto'):
        """
        Split washer into inner circle and outer ring
        
        Args:
            method: 'hough', 'contour', or 'auto' (tries both)
        """
        circles = None
        
        if method == 'hough' or method == 'auto':
            circles = self.detect_circles_hough()
            if circles is not None and len(circles) >= 2:
                print("Using Hough Circle detection method")
            elif method == 'auto':
                circles = None
        
        if circles is None and (method == 'contour' or method == 'auto'):
            contour_circles = self.detect_circles_contours()
            if contour_circles:
                circles = np.array(contour_circles)[:, :3]  # Keep only x, y, radius
                print("Using Contour detection method")
        
        if circles is None or len(circles) < 2:
            print("Could not detect both inner and outer circles!")
            return None, None, None
        
        # Sort circles by radius (smallest first)
        circles_sorted = sorted(circles, key=lambda x: x[2])
        inner_circle = circles_sorted[0]
        outer_circle = circles_sorted[1]
        
        print(f"Inner circle: center=({inner_circle[0]}, {inner_circle[1]}), radius={inner_circle[2]}")
        print(f"Outer circle: center=({outer_circle[0]}, {outer_circle[1]}), radius={outer_circle[2]}")
        
        # Create masks
        inner_mask = np.zeros((self.height, self.width), np.uint8)
        outer_mask = np.zeros((self.height, self.width), np.uint8)
        
        # Draw filled circles on masks
        cv2.circle(inner_mask, (int(inner_circle[0]), int(inner_circle[1])), 
                   int(inner_circle[2]), 255, -1)
        cv2.circle(outer_mask, (int(outer_circle[0]), int(outer_circle[1])), 
                   int(outer_circle[2]), 255, -1)
        
        # Create ring mask (outer minus inner)
        ring_mask = cv2.subtract(outer_mask, inner_mask)
        
        # Apply masks to get separate images
        inner_only = cv2.bitwise_and(self.image, self.image, mask=inner_mask)
        ring_only = cv2.bitwise_and(self.image, self.image, mask=ring_mask)
        
        # Create visualization
        vis_image = self.image.copy()
        cv2.circle(vis_image, (int(inner_circle[0]), int(inner_circle[1])), 
                   int(inner_circle[2]), (0, 255, 0), 2)
        cv2.circle(vis_image, (int(outer_circle[0]), int(outer_circle[1])), 
                   int(outer_circle[2]), (0, 0, 255), 2)
        
        return inner_only, ring_only, vis_image
    
    def save_results(self, output_dir='output'):
        """Save the split images"""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Get base filename
        base_name = Path(self.image_path).stem
        
        # Split the washer
        inner, ring, visualization = self.split_washer()
        
        if inner is None:
            print("Failed to split the washer image!")
            return False
        
        # Save images
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_inner.png'), inner)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_ring.png'), ring)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_visualization.png'), visualization)
        
        print(f"Saved results to {output_dir}/")
        print(f"  - {base_name}_inner.png: Inner circle only")
        print(f"  - {base_name}_ring.png: Outer ring only")
        print(f"  - {base_name}_visualization.png: Detection visualization")
        
        return True
    
    def display_results(self):
        """Display the results in windows"""
        inner, ring, visualization = self.split_washer()
        
        if inner is None:
            print("Failed to split the washer image!")
            return
        
        # Resize for display if images are too large
        max_display_width = 800
        if self.width > max_display_width:
            scale = max_display_width / self.width
            new_width = int(self.width * scale)
            new_height = int(self.height * scale)
            
            inner = cv2.resize(inner, (new_width, new_height))
            ring = cv2.resize(ring, (new_width, new_height))
            visualization = cv2.resize(visualization, (new_width, new_height))
        
        # Display images
        cv2.imshow('Original with Detected Circles', visualization)
        cv2.imshow('Inner Circle Only', inner)
        cv2.imshow('Outer Ring Only', ring)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main function to run the washer splitter"""
    # Configuration
    image_path = input("Enter the path to your washer image: ").strip()
    
    # Remove quotes if present
    image_path = image_path.strip('"').strip("'")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return
    
    # Create splitter instance
    splitter = WasherSplitter(image_path)
    
    # Ask user for action
    print("\nWhat would you like to do?")
    print("1. Display results on screen")
    print("2. Save results to files")
    print("3. Both")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        splitter.display_results()
    
    if choice in ['2', '3']:
        splitter.save_results()
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
