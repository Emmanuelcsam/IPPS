import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class FiberOpticInspector:
    """
    Automated Fiber Optic End Face Defect Detection System
    Based on DO2MR and LEI algorithms from the research paper
    """
    
    def __init__(self, calibration_file: str = "calibration.json"):
        """Initialize the inspector with calibration data"""
        self.calibration = self._load_calibration(calibration_file)
        self.um_per_px = self.calibration.get("um_per_px", 0.7)  # ~0.7 μm/pixel for 10x objective
        
        # Zone definitions (in micrometers)
        self.zones = {
            "core": {"r_min": 0, "r_max": 30, "max_defect_um": 3},
            "cladding": {"r_min": 30, "r_max": 62.5, "max_defect_um": 10},
            "ferrule": {"r_min": 62.5, "r_max": 125, "max_defect_um": 20}
        }
        
        # Detection parameters
        self.do2mr_params = {
            "kernel_size": (15, 15),  # For min-max filtering
            "gamma": 3.0,  # Sensitivity parameter for threshold
            "min_area_px": 30  # Minimum defect area in pixels
        }
        
        self.lei_params = {
            "kernel_size": 15,  # Length of linear detector
            "angles": np.arange(0, 180, 15),  # Detection angles
            "threshold_factor": 2.0  # Threshold multiplier
        }
    
    def _load_calibration(self, filepath: str) -> Dict:
        """Load calibration data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Calibration file not found. Using default values.")
            return {"um_per_px": 0.7}
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the image for defect detection
        Returns: (grayscale, denoised)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise using Gaussian blur
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray, denoised
    
    def find_fiber_center(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Find the center of the fiber using Hough Circle Transform
        """
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find circles using Hough Transform
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=image.shape[0]//4,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=image.shape[0]//2
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Select the most prominent circle (usually the largest)
            largest_circle = circles[0][np.argmax(circles[0][:, 2])]
            return (largest_circle[0], largest_circle[1])
        else:
            # Fallback to image center
            return (image.shape[1]//2, image.shape[0]//2)
    
    def create_zone_masks(self, image_shape: Tuple, center: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Create masks for different zones (core, cladding, ferrule)
        """
        masks = {}
        height, width = image_shape[:2]
        
        # Create coordinate grids
        Y, X = np.ogrid[:height, :width]
        
        # Calculate distance from center
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        for zone_name, zone_params in self.zones.items():
            r_min_px = zone_params["r_min"] / self.um_per_px
            r_max_px = zone_params["r_max"] / self.um_per_px
            
            masks[zone_name] = (dist_from_center >= r_min_px) & (dist_from_center < r_max_px)
        
        return masks
    
    def detect_region_defects_do2mr(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect region-based defects using DO2MR (Difference of Min-Max Ranking)
        Returns: (binary_mask, labeled_defects)
        """
        # Apply min-max ranking filters
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            self.do2mr_params["kernel_size"]
        )
        
        # Maximum filter (dilation)
        img_max = cv2.dilate(image, kernel)
        
        # Minimum filter (erosion)
        img_min = cv2.erode(image, kernel)
        
        # Calculate residual (difference)
        residual = img_max.astype(np.float32) - img_min.astype(np.float32)
        
        # Apply median filter to reduce noise
        residual_filtered = cv2.medianBlur(residual.astype(np.uint8), 3)
        
        # Calculate threshold using sigma-based method
        mean_val = np.mean(residual_filtered)
        std_val = np.std(residual_filtered)
        threshold = mean_val + self.do2mr_params["gamma"] * std_val
        
        # Create binary mask
        _, binary_mask = cv2.threshold(residual_filtered, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological opening to remove small noise
        kernel_open = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Label connected components
        n_labels, labeled = cv2.connectedComponents(binary_mask)
        
        return binary_mask, labeled
    
    def detect_scratches_lei(self, image: np.ndarray) -> np.ndarray:
        """
        Detect scratches using LEI (Linear Enhancement Inspector)
        """
        # Apply histogram equalization for better contrast
        enhanced = cv2.equalizeHist(image)
        
        # Initialize accumulator for all angles
        scratch_strength = np.zeros_like(enhanced, dtype=np.float32)
        
        # Linear detector kernel length
        kernel_length = self.lei_params["kernel_size"]
        
        for angle in self.lei_params["angles"]:
            # Create linear detector at specific angle
            angle_rad = np.deg2rad(angle)
            
            # Generate kernel points along the line
            kernel_points = []
            for i in range(-kernel_length//2, kernel_length//2 + 1):
                x = int(i * np.cos(angle_rad))
                y = int(i * np.sin(angle_rad))
                kernel_points.append((x, y))
            
            # Apply the linear detector
            response = self._apply_linear_detector(enhanced, kernel_points)
            
            # Update maximum response
            scratch_strength = np.maximum(scratch_strength, response)
        
        # Threshold the scratch strength map
        mean_strength = np.mean(scratch_strength)
        std_strength = np.std(scratch_strength)
        threshold = mean_strength + self.lei_params["threshold_factor"] * std_strength
        
        _, scratch_mask = cv2.threshold(scratch_strength, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        scratch_mask = cv2.morphologyEx(scratch_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return scratch_mask
    
    def _apply_linear_detector(self, image: np.ndarray, kernel_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Apply linear detector at specific orientation
        """
        height, width = image.shape
        response = np.zeros_like(image, dtype=np.float32)
        
        # Pad image to handle border effects
        pad_size = len(kernel_points) // 2
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        
        for y in range(height):
            for x in range(width):
                # Calculate response at this pixel
                center_sum = 0
                surround_sum = 0
                valid_points = 0
                
                for dx, dy in kernel_points:
                    px = x + dx + pad_size
                    py = y + dy + pad_size
                    
                    if abs(dx) <= 2 and abs(dy) <= 2:  # Center region
                        center_sum += padded[py, px]
                    else:  # Surrounding region
                        surround_sum += padded[py, px]
                    valid_points += 1
                
                if valid_points > 0:
                    # Calculate difference (center should be darker for scratches)
                    response[y, x] = max(0, surround_sum / (valid_points * 0.7) - center_sum / (valid_points * 0.3))
        
        return response
    
    def classify_defects(self, labeled_image: np.ndarray, scratch_mask: np.ndarray, 
                        zone_masks: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Classify and characterize detected defects
        """
        defects = []
        
        # Process region-based defects
        for label in range(1, labeled_image.max() + 1):
            defect_mask = (labeled_image == label)
            
            # Calculate defect properties
            area_px = np.sum(defect_mask)
            if area_px < self.do2mr_params["min_area_px"]:
                continue
            
            area_um2 = area_px * (self.um_per_px ** 2)
            
            # Find centroid
            y_coords, x_coords = np.where(defect_mask)
            centroid_x = int(np.mean(x_coords))
            centroid_y = int(np.mean(y_coords))
            
            # Determine zone
            zone = "unknown"
            for zone_name, zone_mask in zone_masks.items():
                if zone_mask[centroid_y, centroid_x]:
                    zone = zone_name
                    break
            
            # Calculate bounding box for aspect ratio
            x_min, y_min = np.min(x_coords), np.min(y_coords)
            x_max, y_max = np.max(x_coords), np.max(y_coords)
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = width / height if height > 0 else 1.0
            
            defects.append({
                "type": "dig",
                "zone": zone,
                "area_um2": area_um2,
                "diameter_um": np.sqrt(4 * area_um2 / np.pi),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "aspect_ratio": aspect_ratio
            })
        
        # Process scratches
        scratch_labels, scratch_labeled = cv2.connectedComponents(scratch_mask)
        
        for label in range(1, scratch_labels):
            scratch_region = (scratch_labeled == label)
            area_px = np.sum(scratch_region)
            
            if area_px < 10:  # Minimum scratch size
                continue
            
            # Find scratch properties
            y_coords, x_coords = np.where(scratch_region)
            
            # Fit line to scratch
            if len(x_coords) > 5:
                vx, vy, x0, y0 = cv2.fitLine(
                    np.column_stack([x_coords, y_coords]),
                    cv2.DIST_L2, 0, 0.01, 0.01
                )
                
                # Calculate length
                points = np.column_stack([x_coords, y_coords])
                distances = np.sqrt(np.sum((points - [x0, y0])**2, axis=1))
                length_px = np.max(distances) * 2
                length_um = length_px * self.um_per_px
                
                # Determine zone
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))
                
                zone = "unknown"
                for zone_name, zone_mask in zone_masks.items():
                    if zone_mask[centroid_y, centroid_x]:
                        zone = zone_name
                        break
                
                defects.append({
                    "type": "scratch",
                    "zone": zone,
                    "length_um": length_um,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "aspect_ratio": length_px / np.sqrt(area_px)
                })
        
        return pd.DataFrame(defects)
    
    def apply_pass_fail_criteria(self, defects_df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Apply IEC-61300 based pass/fail criteria
        """
        status = "PASS"
        failure_reasons = []
        
        for zone_name, zone_params in self.zones.items():
            zone_defects = defects_df[defects_df["zone"] == zone_name]
            
            if len(zone_defects) == 0:
                continue
            
            # Check dig sizes
            digs = zone_defects[zone_defects["type"] == "dig"]
            if len(digs) > 0:
                max_dig_diameter = digs["diameter_um"].max()
                if max_dig_diameter > zone_params["max_defect_um"]:
                    status = "FAIL"
                    failure_reasons.append(
                        f"{zone_name}: Dig diameter {max_dig_diameter:.1f}μm exceeds limit {zone_params['max_defect_um']}μm"
                    )
            
            # Check scratch lengths
            scratches = zone_defects[zone_defects["type"] == "scratch"]
            if len(scratches) > 0:
                max_scratch_length = scratches["length_um"].max()
                if max_scratch_length > zone_params["max_defect_um"]:
                    status = "FAIL"
                    failure_reasons.append(
                        f"{zone_name}: Scratch length {max_scratch_length:.1f}μm exceeds limit {zone_params['max_defect_um']}μm"
                    )
            
            # Check total defect count (example: max 5 defects in core)
            if zone_name == "core" and len(zone_defects) > 5:
                status = "FAIL"
                failure_reasons.append(f"Core: Too many defects ({len(zone_defects)} > 5)")
        
        return status, failure_reasons
    
    def inspect_fiber(self, image_path: str) -> Dict:
        """
        Main inspection function - complete pipeline
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess
        gray, denoised = self.preprocess_image(image)
        
        # Find fiber center
        center = self.find_fiber_center(denoised)
        
        # Create zone masks
        zone_masks = self.create_zone_masks(gray.shape, center)
        
        # Detect region-based defects using DO2MR
        region_mask, labeled_regions = self.detect_region_defects_do2mr(denoised)
        
        # Detect scratches using LEI
        scratch_mask = self.detect_scratches_lei(gray)
        
        # Classify defects
        defects_df = self.classify_defects(labeled_regions, scratch_mask, zone_masks)
        
        # Apply pass/fail criteria
        status, failure_reasons = self.apply_pass_fail_criteria(defects_df)
        
        # Prepare results
        results = {
            "status": status,
            "failure_reasons": failure_reasons,
            "defect_count": len(defects_df),
            "defects": defects_df.to_dict('records'),
            "fiber_center": center,
            "masks": {
                "region_defects": region_mask,
                "scratches": scratch_mask,
                "zones": zone_masks
            }
        }
        
        return results
    
    def visualize_results(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        """
        Visualize inspection results
        """
        # Load original image
        image = cv2.imread(image_path)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create visualization
        vis_image = image.copy()
        
        # Draw zone boundaries
        center = results["fiber_center"]
        for zone_name, zone_params in self.zones.items():
            radius_px = int(zone_params["r_max"] / self.um_per_px)
            color = {"core": (255, 0, 0), "cladding": (0, 255, 0), "ferrule": (0, 0, 255)}[zone_name]
            cv2.circle(vis_image, center, radius_px, color, 2)
        
        # Overlay defects
        region_mask = results["masks"]["region_defects"]
        scratch_mask = results["masks"]["scratches"]
        
        # Color defects
        vis_image[region_mask > 0] = [0, 255, 255]  # Yellow for digs
        vis_image[scratch_mask > 0] = [255, 0, 255]  # Magenta for scratches
        
        # Add status text
        status = results["status"]
        color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        cv2.putText(vis_image, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add defect count
        cv2.putText(vis_image, f"Defects: {results['defect_count']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Region defects
        axes[0, 1].imshow(region_mask, cmap='hot')
        axes[0, 1].set_title("Region-based Defects (DO2MR)")
        axes[0, 1].axis('off')
        
        # Scratches
        axes[1, 0].imshow(scratch_mask, cmap='hot')
        axes[1, 0].set_title("Scratches (LEI)")
        axes[1, 0].axis('off')
        
        # Final result
        axes[1, 1].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"Final Result: {status}")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        return vis_image


# Calibration function
def calibrate_system(calibration_image_path: str, dot_spacing_um: float = 10.0) -> float:
    """
    Calibrate the system using a dot grid calibration target
    """
    image = cv2.imread(calibration_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load calibration image: {calibration_image_path}")
    
    # Threshold to find dots
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours (dots)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate centroids
    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30:  # Filter small noise
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroids.append((cx, cy))
    
    if len(centroids) < 2:
        raise ValueError("Not enough calibration dots found")
    
    # Sort centroids by x-coordinate
    centroids = sorted(centroids, key=lambda x: x[0])
    
    # Calculate average spacing in pixels
    distances = []
    for i in range(1, len(centroids)):
        dist = np.sqrt((centroids[i][0] - centroids[i-1][0])**2 + 
                      (centroids[i][1] - centroids[i-1][1])**2)
        distances.append(dist)
    
    avg_distance_px = np.mean(distances)
    um_per_px = dot_spacing_um / avg_distance_px
    
    # Save calibration
    calibration = {"um_per_px": um_per_px}
    with open("calibration.json", "w") as f:
        json.dump(calibration, f)
    
    print(f"Calibration complete: {um_per_px:.3f} μm/pixel")
    return um_per_px


# Example usage
if __name__ == "__main__":
    # First, calibrate if needed
    # um_per_px = calibrate_system("calibration_target.png", dot_spacing_um=10.0)
    base_path = 'C:/Users/Saem1001/Documents/GitHub/OpenCV-Practice/'
    img_path = base_path + 'img1.jpg' 
    # Create inspector instance
    inspector = FiberOpticInspector()
    
    # Inspect a fiber end face
    results = inspector.inspect_fiber("img_path")
    
    # Print results
    print(f"Inspection Status: {results['status']}")
    print(f"Total defects found: {results['defect_count']}")
    
    if results['failure_reasons']:
        print("\nFailure reasons:")
        for reason in results['failure_reasons']:
            print(f"  - {reason}")
    
    # Display defect details
    if results['defect_count'] > 0:
        print("\nDefect details:")
        defects_df = pd.DataFrame(results['defects'])
        print(defects_df)
    
    # Visualize results
    inspector.visualize_results("img_path", results, "img_path")

