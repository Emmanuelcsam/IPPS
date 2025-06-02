import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

class ComprehensiveImageProcessor:
    """
    A comprehensive image processor implementing all OpenCV techniques from the documentation:
    - Hough Circle Transform
    - Hough Line Transform
    - Canny Edge Detection
    - Color Space Conversion
    - Morphological Transformations
    - Histograms
    - Image Gradients
    - Image Segmentation (Watershed)
    - Image Thresholding
    - Foreground Extraction (GrabCut)
    - Template Matching
    """
    
    def __init__(self, image_path):
        """Initialize with an image path"""
        self.original_image = cv.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        self.gray_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
        print(f"Image loaded successfully: {image_path}")
        print(f"Image shape: {self.original_image.shape}")
    
    def hough_circle_detection(self):
        """Detect circles using Hough Circle Transform"""
        print("\n--- Hough Circle Detection ---")
        
        # Apply median blur to reduce noise
        img = cv.medianBlur(self.gray_image, 5)
        
        # Detect circles
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20,
                                param1=50, param2=30, minRadius=10, maxRadius=100)
        
        # Draw circles
        result = self.original_image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw center
                cv.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
            print(f"Detected {len(circles[0])} circles")
        else:
            print("No circles detected")
        
        return result
    
    def hough_line_detection(self):
        """Detect lines using Hough Line Transform"""
        print("\n--- Hough Line Detection ---")
        
        # Edge detection
        edges = cv.Canny(self.gray_image, 50, 150, apertureSize=3)
        
        # Standard Hough Line Transform
        lines = cv.HoughLines(edges, 1, np.pi/180, 100)
        result_standard = self.original_image.copy()
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(result_standard, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Probabilistic Hough Line Transform
        linesP = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        result_prob = self.original_image.copy()
        
        if linesP is not None:
            for line in linesP:
                x1, y1, x2, y2 = line[0]
                cv.line(result_prob, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print(f"Detected {len(linesP)} line segments")
        
        return result_standard, result_prob
    
    def canny_edge_detection(self, low_threshold=100, high_threshold=200):
        """Apply Canny edge detection"""
        print("\n--- Canny Edge Detection ---")
        edges = cv.Canny(self.gray_image, low_threshold, high_threshold)
        print(f"Edge detection with thresholds: {low_threshold}, {high_threshold}")
        return edges
    
    def color_space_conversion(self):
        """Convert between color spaces"""
        print("\n--- Color Space Conversion ---")
        
        # BGR to HSV
        hsv = cv.cvtColor(self.original_image, cv.COLOR_BGR2HSV)
        
        # Extract blue objects (example)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        blue_objects = cv.bitwise_and(self.original_image, self.original_image, mask=mask)
        
        return hsv, mask, blue_objects
    
    def morphological_operations(self):
        """Apply morphological transformations"""
        print("\n--- Morphological Operations ---")
        
        # Create binary image for morphological operations
        _, binary = cv.threshold(self.gray_image, 127, 255, cv.THRESH_BINARY)
        
        # Define kernel
        kernel = np.ones((5, 5), np.uint8)
        
        # Apply different morphological operations
        erosion = cv.erode(binary, kernel, iterations=1)
        dilation = cv.dilate(binary, kernel, iterations=1)
        opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
        gradient = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)
        tophat = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)
        blackhat = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)
        
        print("Applied all morphological operations")
        
        return {
            'erosion': erosion,
            'dilation': dilation,
            'opening': opening,
            'closing': closing,
            'gradient': gradient,
            'tophat': tophat,
            'blackhat': blackhat
        }
    
    def histogram_analysis(self):
        """Calculate and analyze histograms"""
        print("\n--- Histogram Analysis ---")
        
        # Calculate histogram for grayscale
        hist_gray = cv.calcHist([self.gray_image], [0], None, [256], [0, 256])
        
        # Calculate histogram for each BGR channel
        colors = ('b', 'g', 'r')
        hist_bgr = {}
        for i, col in enumerate(colors):
            hist_bgr[col] = cv.calcHist([self.original_image], [i], None, [256], [0, 256])
        
        # Apply histogram with mask
        mask = np.zeros(self.gray_image.shape[:2], np.uint8)
        h, w = self.gray_image.shape[:2]
        mask[h//4:3*h//4, w//4:3*w//4] = 255
        masked_img = cv.bitwise_and(self.gray_image, self.gray_image, mask=mask)
        hist_mask = cv.calcHist([self.gray_image], [0], mask, [256], [0, 256])
        
        print("Calculated histograms for all channels")
        
        return hist_gray, hist_bgr, hist_mask, mask
    
    def image_gradients(self):
        """Calculate image gradients using Sobel, Scharr, and Laplacian"""
        print("\n--- Image Gradients ---")
        
        # Sobel gradients
        sobelx = cv.Sobel(self.gray_image, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(self.gray_image, cv.CV_64F, 0, 1, ksize=5)
        
        # Handle negative slopes properly
        sobelx_abs = np.absolute(sobelx)
        sobely_abs = np.absolute(sobely)
        sobelx_8u = np.uint8(sobelx_abs)
        sobely_8u = np.uint8(sobely_abs)
        
        # Laplacian
        laplacian = cv.Laplacian(self.gray_image, cv.CV_64F)
        laplacian_abs = np.absolute(laplacian)
        laplacian_8u = np.uint8(laplacian_abs)
        
        print("Calculated Sobel and Laplacian gradients")
        
        return {
            'sobelx': sobelx_8u,
            'sobely': sobely_8u,
            'laplacian': laplacian_8u
        }
    
    def watershed_segmentation(self):
        """Apply watershed algorithm for image segmentation"""
        print("\n--- Watershed Segmentation ---")
        
        # Threshold
        _, thresh = cv.threshold(self.gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        result = self.original_image.copy()
        markers = cv.watershed(result, markers)
        result[markers == -1] = [255, 0, 0]
        
        print("Applied watershed segmentation")
        
        return result, markers
    
    def thresholding_operations(self):
        """Apply various thresholding techniques"""
        print("\n--- Thresholding Operations ---")
        
        # Simple thresholding
        _, binary = cv.threshold(self.gray_image, 127, 255, cv.THRESH_BINARY)
        _, binary_inv = cv.threshold(self.gray_image, 127, 255, cv.THRESH_BINARY_INV)
        _, trunc = cv.threshold(self.gray_image, 127, 255, cv.THRESH_TRUNC)
        _, tozero = cv.threshold(self.gray_image, 127, 255, cv.THRESH_TOZERO)
        _, tozero_inv = cv.threshold(self.gray_image, 127, 255, cv.THRESH_TOZERO_INV)
        
        # Adaptive thresholding
        adaptive_mean = cv.adaptiveThreshold(self.gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                           cv.THRESH_BINARY, 11, 2)
        adaptive_gaussian = cv.adaptiveThreshold(self.gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv.THRESH_BINARY, 11, 2)
        
        # Otsu's thresholding
        _, otsu = cv.threshold(self.gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        print("Applied all thresholding techniques")
        
        return {
            'binary': binary,
            'binary_inv': binary_inv,
            'trunc': trunc,
            'tozero': tozero,
            'tozero_inv': tozero_inv,
            'adaptive_mean': adaptive_mean,
            'adaptive_gaussian': adaptive_gaussian,
            'otsu': otsu
        }
    
    def grabcut_segmentation(self, rect=None):
        """Apply GrabCut algorithm for foreground extraction"""
        print("\n--- GrabCut Segmentation ---")
        
        mask = np.zeros(self.gray_image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # If no rect provided, use center region
        if rect is None:
            h, w = self.gray_image.shape[:2]
            rect = (w//4, h//4, w//2, h//2)
        
        # Apply GrabCut
        cv.grabCut(self.original_image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        
        # Modify mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = self.original_image * mask2[:, :, np.newaxis]
        
        print(f"Applied GrabCut with rect: {rect}")
        
        return result, mask2
    
    def template_matching(self, template_path=None):
        """Perform template matching"""
        print("\n--- Template Matching ---")
        
        # If no template provided, use a region from the image itself
        if template_path is None:
            h, w = self.gray_image.shape[:2]
            template = self.gray_image[h//4:h//2, w//4:w//2]
        else:
            template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
        
        if template is None:
            print("Could not load template")
            return None
        
        # Apply template matching
        methods = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF_NORMED']
        results = {}
        
        for method_name in methods:
            method = eval(method_name)
            res = cv.matchTemplate(self.gray_image, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            
            # Get the correct location
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            
            # Draw rectangle
            result = self.original_image.copy()
            h, w = template.shape
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(result, top_left, bottom_right, 255, 2)
            
            results[method_name] = result
            print(f"Applied {method_name}")
        
        return results
    
    def visualize_results(self, results_dict, title="Results"):
        """Visualize multiple results in a grid"""
        n_results = len(results_dict)
        cols = 3
        rows = (n_results + cols - 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        
        for i, (name, img) in enumerate(results_dict.items()):
            plt.subplot(rows, cols, i + 1)
            
            # Handle different image types
            if len(img.shape) == 2:  # Grayscale
                plt.imshow(img, cmap='gray')
            else:  # Color
                plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            
            plt.title(name)
            plt.xticks([])
            plt.yticks([])
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def process_all(self):
        """Run all image processing techniques"""
        print("\n=== Running All Image Processing Techniques ===\n")
        
        # Store all results
        all_results = {}
        
        # 1. Hough Circle Detection
        circles = self.hough_circle_detection()
        all_results['Hough Circles'] = circles
        
        # 2. Hough Line Detection
        lines_standard, lines_prob = self.hough_line_detection()
        all_results['Hough Lines (Standard)'] = lines_standard
        all_results['Hough Lines (Probabilistic)'] = lines_prob
        
        # 3. Canny Edge Detection
        edges = self.canny_edge_detection()
        all_results['Canny Edges'] = edges
        
        # 4. Color Space Conversion
        hsv, mask, blue_objects = self.color_space_conversion()
        all_results['HSV'] = hsv
        all_results['Blue Mask'] = mask
        all_results['Blue Objects'] = blue_objects
        
        # 5. Morphological Operations
        morph_results = self.morphological_operations()
        for name, result in morph_results.items():
            all_results[f'Morph {name}'] = result
        
        # 6. Image Gradients
        gradient_results = self.image_gradients()
        for name, result in gradient_results.items():
            all_results[f'Gradient {name}'] = result
        
        # 7. Watershed Segmentation
        watershed_result, markers = self.watershed_segmentation()
        all_results['Watershed'] = watershed_result
        
        # 8. Thresholding
        thresh_results = self.thresholding_operations()
        for name, result in thresh_results.items():
            all_results[f'Thresh {name}'] = result
        
        # 9. GrabCut
        grabcut_result, grabcut_mask = self.grabcut_segmentation()
        all_results['GrabCut'] = grabcut_result
        
        # 10. Template Matching
        template_results = self.template_matching()
        if template_results:
            for name, result in template_results.items():
                all_results[f'Template {name}'] = result
        
        print("\n=== All processing complete! ===")
        return all_results


# Main execution function
def main():
    # Process the three uploaded images
    base_path = 'C:/Users/Saem1001/Documents/GitHub/OpenCV-Practice/'
    image_files = [base_path + "image1.jpg", "image2.jpg", "image3.jpg"]
    
    for img_file in image_files:
        print(f"\n{'='*60}")
        print(f"Processing: {img_file}")
        print(f"{'='*60}")
        
        try:
            # Create processor instance
            processor = ComprehensiveImageProcessor(img_file)
            
            # Process all techniques
            results = processor.process_all()
            
            # Visualize results
            processor.visualize_results(results, title=f"All Processing Results - {img_file}")
            
            # Save some key results
            cv.imwrite(f"edges_{img_file}", results['Canny Edges'])
            cv.imwrite(f"circles_{img_file}", results['Hough Circles'])
            cv.imwrite(f"watershed_{img_file}", results['Watershed'])
            cv.imwrite(f"grabcut_{img_file}", results['GrabCut'])
            
            print(f"\nResults saved for {img_file}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    print("\n\nAll processing complete!")


if __name__ == "__main__":
    main()