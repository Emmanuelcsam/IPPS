import numpy as np
import cv2 as cv
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

@dataclass
class FiberOpticAnalysis:
    """Store fiber optic analysis results"""
    core_center: Tuple[int, int]
    core_radius: float
    cladding_radius: float
    ferrule_radius: float
    core_area: float
    cladding_area: float
    core_intensity: float
    cladding_intensity: float

def preprocess_fiber_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced preprocessing for fiber optic images"""
    # Apply Gaussian blur to reduce noise
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    
    # Apply CLAHE for better contrast
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Create edge map for better boundary detection
    edges = cv.Canny(enhanced, 50, 150)
    
    return enhanced, edges

def find_fiber_center_and_boundaries(img: np.ndarray) -> Tuple[Tuple[int, int], List[float]]:
    """Find fiber center using intensity-based approach"""
    # Find the brightest region (typically the core)
    # Apply threshold to isolate bright regions
    _, bright_mask = cv.threshold(img, np.percentile(img, 90), 255, cv.THRESH_BINARY)
    
    # Find contours of bright regions
    contours, _ = cv.findContours(bright_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to image center
        h, w = img.shape
        return (w//2, h//2), []
    
    # Find the largest bright region (should be the core)
    largest_contour = max(contours, key=cv.contourArea)
    M = cv.moments(largest_contour)
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        h, w = img.shape
        cx, cy = w//2, h//2
    
    return (cx, cy), []

def analyze_radial_intensity_gradient(img: np.ndarray, center: Tuple[int, int], 
                                    max_radius: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyze radial intensity profile and its gradient"""
    cy, cx = center
    radii = np.arange(0, max_radius)
    intensities = []
    
    for r in radii:
        if r == 0:
            intensities.append(float(img[cy, cx]))
            continue
            
        # Sample points in a circle with more samples for larger radii
        num_samples = max(16, int(2 * np.pi * r))
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        
        x_coords = cx + r * np.cos(angles)
        y_coords = cy + r * np.sin(angles)
        
        # Ensure coordinates are within image bounds
        valid_mask = (x_coords >= 0) & (x_coords < img.shape[1]) & \
                     (y_coords >= 0) & (y_coords < img.shape[0])
        
        if np.any(valid_mask):
            x_coords = x_coords[valid_mask].astype(int)
            y_coords = y_coords[valid_mask].astype(int)
            intensities.append(float(np.mean(img[y_coords, x_coords])))
        else:
            intensities.append(0.0)
    
    intensities = np.array(intensities)
    
    # Smooth the intensity profile
    smoothed = gaussian_filter1d(intensities, sigma=3)
    
    # Calculate gradient
    gradient = np.gradient(smoothed)
    
    return radii, intensities, gradient

def detect_fiber_boundaries(radii: np.ndarray, intensities: np.ndarray, 
                          gradient: np.ndarray) -> Tuple[int, int, int]:
    """Detect core, cladding, and ferrule boundaries using multi-scale analysis"""
    
    # Normalize gradient
    grad_abs = np.abs(gradient)
    if np.max(grad_abs) > 0:
        grad_normalized = grad_abs / np.max(grad_abs)
    else:
        grad_normalized = grad_abs
    
    # Find peaks in gradient (boundaries)
    # Use different prominence thresholds for different features
    peaks_high, properties_high = find_peaks(grad_normalized, prominence=0.3, distance=5)
    peaks_low, properties_low = find_peaks(grad_normalized, prominence=0.1, distance=5)
    
    # Combine and sort peaks
    all_peaks = np.unique(np.concatenate([peaks_high, peaks_low]))
    all_peaks = all_peaks[all_peaks > 5]  # Ignore very small radii
    
    # Analyze intensity drops
    intensity_drops = []
    for i in range(1, len(intensities)-5):
        drop = (intensities[i-1] - intensities[i+5]) / (intensities[i-1] + 1e-6)
        intensity_drops.append(drop)
    intensity_drops = np.array(intensity_drops)
    
    # Identify boundaries based on both gradient and intensity drops
    core_radius = None
    cladding_radius = None
    ferrule_radius = None
    
    # Find core boundary (first significant intensity drop)
    for peak in all_peaks:
        if peak < len(intensity_drops) and intensity_drops[peak] > 0.2:
            core_radius = peak
            break
    
    # If no core found using drops, use first gradient peak
    if core_radius is None and len(all_peaks) > 0:
        core_radius = all_peaks[0]
    
    # Find cladding boundary (next significant change after core)
    if core_radius is not None:
        cladding_candidates = all_peaks[all_peaks > core_radius * 2]
        if len(cladding_candidates) > 0:
            # Look for the peak with significant intensity change
            for candidate in cladding_candidates:
                if candidate < len(intensities) - 1:
                    local_drop = (intensities[core_radius] - intensities[candidate]) / (intensities[core_radius] + 1e-6)
                    if local_drop > 0.3:
                        cladding_radius = candidate
                        break
            
            # If no good candidate, use the first one
            if cladding_radius is None:
                cladding_radius = cladding_candidates[0]
    
    # Find ferrule boundary (large structure)
    if cladding_radius is not None:
        ferrule_candidates = all_peaks[all_peaks > cladding_radius * 1.5]
        if len(ferrule_candidates) > 0:
            ferrule_radius = ferrule_candidates[0]
    
    # Apply standard fiber ratios if detection seems off
    if core_radius and cladding_radius:
        ratio = core_radius / cladding_radius
        # Standard single-mode fiber has core/cladding ratio of about 0.07 (9/125)
        # Standard multimode fiber has ratio of about 0.4 (50/125) or 0.5 (62.5/125)
        if ratio > 0.7:  # Likely misdetection
            # Assume we detected cladding as core
            actual_core = int(cladding_radius * 0.4)  # Assume 50/125 fiber
            if actual_core in all_peaks:
                core_radius = actual_core
            else:
                # Find closest peak
                distances = np.abs(all_peaks - actual_core)
                if len(distances) > 0:
                    core_radius = all_peaks[np.argmin(distances)]
    
    # Set defaults if not found
    if core_radius is None:
        core_radius = len(radii) // 10
    if cladding_radius is None:
        cladding_radius = int(core_radius * 2.5)
    if ferrule_radius is None:
        ferrule_radius = int(cladding_radius * 2)
    
    return int(core_radius), int(cladding_radius), int(ferrule_radius)

def refine_detection_with_circles(img: np.ndarray, center: Tuple[int, int], 
                                 initial_core: int, initial_cladding: int) -> Tuple[int, int]:
    """Refine boundary detection using Hough circles"""
    # Look for circles near the expected radii
    circles_core = cv.HoughCircles(
        img,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=int(initial_core * 0.7),
        maxRadius=int(initial_core * 1.3)
    )
    
    circles_cladding = cv.HoughCircles(
        img,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=20,
        minRadius=int(initial_cladding * 0.7),
        maxRadius=int(initial_cladding * 1.3)
    )
    
    refined_core = initial_core
    refined_cladding = initial_cladding
    
    # Refine core
    if circles_core is not None:
        # Find circle closest to center
        min_dist = float('inf')
        for circle in circles_core[0]:
            dist = np.sqrt((circle[0] - center[0])**2 + (circle[1] - center[1])**2)
            if dist < min_dist:
                min_dist = dist
                refined_core = int(circle[2])
    
    # Refine cladding
    if circles_cladding is not None:
        # Find circle closest to center
        min_dist = float('inf')
        for circle in circles_cladding[0]:
            dist = np.sqrt((circle[0] - center[0])**2 + (circle[1] - center[1])**2)
            if dist < min_dist:
                min_dist = dist
                refined_cladding = int(circle[2])
    
    return refined_core, refined_cladding

def analyze_fiber_optic_enhanced(img: np.ndarray) -> Optional[FiberOpticAnalysis]:
    """Enhanced fiber optic analysis for difficult images"""
    # Preprocess
    processed, edges = preprocess_fiber_image(img)
    
    # Find center
    center, _ = find_fiber_center_and_boundaries(processed)
    
    # Analyze radial profile
    max_radius = min(center[0], center[1], img.shape[1]-center[0], img.shape[0]-center[1])
    radii, intensities, gradient = analyze_radial_intensity_gradient(processed, center, max_radius)
    
    # Detect boundaries
    core_radius, cladding_radius, ferrule_radius = detect_fiber_boundaries(radii, intensities, gradient)
    
    # Refine with circle detection
    core_radius, cladding_radius = refine_detection_with_circles(processed, center, core_radius, cladding_radius)
    
    # Calculate metrics
    core_area = np.pi * core_radius**2
    cladding_area = np.pi * cladding_radius**2
    
    # Calculate average intensities using masks
    core_mask = np.zeros(img.shape, dtype=np.uint8)
    cv.circle(core_mask, center, core_radius, 255, -1)
    core_intensity = np.mean(img[core_mask > 0]) if np.any(core_mask > 0) else 0
    
    cladding_mask = np.zeros(img.shape, dtype=np.uint8)
    cv.circle(cladding_mask, center, cladding_radius, 255, -1)
    cv.circle(cladding_mask, center, core_radius, 0, -1)
    cladding_intensity = np.mean(img[cladding_mask > 0]) if np.any(cladding_mask > 0) else 0
    
    return FiberOpticAnalysis(
        core_center=center,
        core_radius=core_radius,
        cladding_radius=cladding_radius,
        ferrule_radius=ferrule_radius,
        core_area=core_area,
        cladding_area=cladding_area,
        core_intensity=core_intensity,
        cladding_intensity=cladding_intensity
    )

def visualize_fiber_analysis_enhanced(img: np.ndarray, analysis: FiberOpticAnalysis, 
                                    radii: np.ndarray, intensities: np.ndarray) -> np.ndarray:
    """Enhanced visualization with profile overlay"""
    # Convert to color
    if len(img.shape) == 2:
        vis_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        vis_img = img.copy()
    
    # Draw core (red) - thicker line
    cv.circle(vis_img, analysis.core_center, int(analysis.core_radius), (0, 0, 255), 3)
    
    # Draw cladding (green)
    cv.circle(vis_img, analysis.core_center, int(analysis.cladding_radius), (0, 255, 0), 2)
    
    # Draw ferrule/outer boundary (yellow) - dashed approximation
    if analysis.ferrule_radius < min(vis_img.shape[:2]) // 2:
        # Draw dotted circle
        num_segments = 40
        for i in range(num_segments):
            angle1 = 2 * np.pi * i / num_segments
            angle2 = 2 * np.pi * (i + 0.5) / num_segments
            pt1 = (int(analysis.core_center[0] + analysis.ferrule_radius * np.cos(angle1)),
                   int(analysis.core_center[1] + analysis.ferrule_radius * np.sin(angle1)))
            pt2 = (int(analysis.core_center[0] + analysis.ferrule_radius * np.cos(angle2)),
                   int(analysis.core_center[1] + analysis.ferrule_radius * np.sin(angle2)))
            cv.line(vis_img, pt1, pt2, (0, 255, 255), 1)
    
    # Draw center crosshair
    cv.line(vis_img, (analysis.core_center[0]-10, analysis.core_center[1]), 
            (analysis.core_center[0]+10, analysis.core_center[1]), (255, 0, 0), 2)
    cv.line(vis_img, (analysis.core_center[0], analysis.core_center[1]-10), 
            (analysis.core_center[0], analysis.core_center[1]+10), (255, 0, 0), 2)
    
    # Add text annotations with background
    font = cv.FONT_HERSHEY_SIMPLEX
    
    # Create text with background for better visibility
    def put_text_with_background(img, text, pos, color):
        (text_width, text_height), _ = cv.getTextSize(text, font, 0.6, 1)
        cv.rectangle(img, (pos[0]-5, pos[1]-text_height-5), 
                    (pos[0]+text_width+5, pos[1]+5), (0, 0, 0), -1)
        cv.putText(img, text, pos, font, 0.6, color, 1)
    
    put_text_with_background(vis_img, f"Core: {analysis.core_radius:.0f}px", 
                            (10, 30), (0, 0, 255))
    put_text_with_background(vis_img, f"Cladding: {analysis.cladding_radius:.0f}px", 
                            (10, 60), (0, 255, 0))
    put_text_with_background(vis_img, f"Ratio: {analysis.core_radius/analysis.cladding_radius:.3f}", 
                            (10, 90), (255, 255, 255))
    
    return vis_img

# Main execution
base_path = 'C:/Users/Saem1001/Documents/GitHub/OpenCV-Practice/'
image_path = base_path + 'img1.jpg'

if not os.path.exists(image_path):
    print(f"Error: File could not be read. Check if '{image_path}' exists.")
else:
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image at path '{image_path}' could not be loaded.")
    else:
        print("Analyzing fiber optic end face...")
        
        # Perform analysis
        analysis = analyze_fiber_optic_enhanced(img)
        
        if analysis:
            print("\n=== Fiber Optic Analysis Results ===")
            print(f"Core Center: {analysis.core_center}")
            print(f"Core Radius: {analysis.core_radius:.1f} pixels")
            print(f"Cladding Radius: {analysis.cladding_radius:.1f} pixels")
            print(f"Core/Cladding Ratio: {analysis.core_radius/analysis.cladding_radius:.3f}")
            
            # Determine fiber type based on ratio
            ratio = analysis.core_radius/analysis.cladding_radius
            if ratio < 0.1:
                fiber_type = "Single-mode (9/125 μm)"
            elif 0.35 < ratio < 0.45:
                fiber_type = "Multimode (50/125 μm)"
            elif 0.45 < ratio < 0.55:
                fiber_type = "Multimode (62.5/125 μm)"
            else:
                fiber_type = "Unknown"
            
            print(f"Probable Fiber Type: {fiber_type}")
            print(f"Core Intensity: {analysis.core_intensity:.1f}")
            print(f"Cladding Intensity: {analysis.cladding_intensity:.1f}")
            
            # Get radial profile for visualization
            center = analysis.core_center
            max_radius = min(center[0], center[1], img.shape[1]-center[0], img.shape[0]-center[1])
            radii, intensities, gradient = analyze_radial_intensity_gradient(img, center, max_radius)
            
            # Visualize results
            result_img = visualize_fiber_analysis_enhanced(img, analysis, radii, intensities)
            
            # Create comparison view
            comparison = np.hstack([
                cv.cvtColor(img, cv.COLOR_GRAY2BGR),
                result_img
            ])
            
            # Add labels
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
            cv.putText(comparison, "Analysis", (img.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
            
            cv.imshow('Fiber Optic Analysis', comparison)
            
            # Create detailed plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot intensity profile
            ax1.plot(radii[:len(intensities)], intensities, 'b-', linewidth=2, label='Intensity')
            ax1.axvline(x=analysis.core_radius, color='r', linestyle='--', 
                       label=f'Core ({analysis.core_radius:.0f}px)', linewidth=2)
            ax1.axvline(x=analysis.cladding_radius, color='g', linestyle='--', 
                       label=f'Cladding ({analysis.cladding_radius:.0f}px)', linewidth=2)
            ax1.set_xlabel('Radius (pixels)')
            ax1.set_ylabel('Intensity')
            ax1.set_title('Radial Intensity Profile')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, min(analysis.cladding_radius * 2, len(radii)-1))
            
            # Plot gradient
            gradient_smooth = gaussian_filter1d(np.abs(gradient), sigma=2)
            ax2.plot(radii[:len(gradient)], gradient_smooth, 'r-', linewidth=2)
            ax2.axvline(x=analysis.core_radius, color='r', linestyle='--', linewidth=2)
            ax2.axvline(x=analysis.cladding_radius, color='g', linestyle='--', linewidth=2)
            ax2.set_xlabel('Radius (pixels)')
            ax2.set_ylabel('|Gradient|')
            ax2.set_title('Intensity Gradient (Boundary Detection)')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, min(analysis.cladding_radius * 2, len(radii)-1))
            
            plt.tight_layout()
            plt.show()
            
            print("\nPress any key while the image window is active to close it.")
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Analysis failed. Please check the image quality.")

