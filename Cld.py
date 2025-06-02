import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

class ComprehensiveImageProcessor:
    def __init__(self, image_paths):
        """Initialize with list of image paths"""
        self.image_paths = image_paths
        self.images = []
        self.load_images()
        
    def load_images(self):
        """Load all images"""
        for path in self.image_paths:
            if os.path.exists(path):
                img = cv.imread(path)
                if img is not None:
                    self.images.append(img)
                    print(f"Successfully loaded: {path}")
                else:
                    print(f"Failed to load: {path}")
            else:
                print(f"File not found: {path}")
    
    def canny_edge_detection(self, img, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, low_threshold, high_threshold)
        return edges
    
    def hough_line_detection(self, img):
        """Apply Hough line detection"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150, apertureSize=3)
        
        # Standard Hough Transform
        lines = cv.HoughLines(edges, 1, np.pi/180, threshold=100)
        img_lines = img.copy()
        
        if lines is not None:
            for line in lines[:10]:  # Limit to first 10 lines
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Probabilistic Hough Transform
        lines_p = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        img_lines_p = img.copy()
        
        if lines_p is not None:
            for line in lines_p:
                x1, y1, x2, y2 = line[0]
                cv.line(img_lines_p, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return img_lines, img_lines_p
    
    def hough_circle_detection(self, img):
        """Apply Hough circle detection"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.medianBlur(gray, 5)
        
        circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, 20,
                                param1=50, param2=30, minRadius=0, maxRadius=0)
        
        img_circles = img.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv.circle(img_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw center
                cv.circle(img_circles, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        return img_circles
    
    def color_space_conversion_and_tracking(self, img):
        """Convert color spaces and demonstrate object tracking"""
        # BGR to HSV conversion
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        # BGR to Grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Object tracking - track green objects
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask_green = cv.inRange(hsv, lower_green, upper_green)
        result_green = cv.bitwise_and(img, img, mask=mask_green)
        
        return hsv, gray, mask_green, result_green
    
    def calculate_histogram(self, img):
        """Calculate and plot histogram"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate color histogram for BGR channels
        colors = ('b', 'g', 'r')
        hist_color = []
        for i, color in enumerate(colors):
            hist_c = cv.calcHist([img], [i], None, [256], [0, 256])
            hist_color.append(hist_c)
        
        return hist, hist_color
    
    def image_gradients(self, img):
        """Apply Sobel, Scharr, and Laplacian operators"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Laplacian
        laplacian = cv.Laplacian(gray, cv.CV_64F)
        
        # Sobel X and Y
        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
        
        # Convert to uint8
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        
        sobelx = np.absolute(sobelx)
        sobelx = np.uint8(sobelx)
        
        sobely = np.absolute(sobely)
        sobely = np.uint8(sobely)
        
        return laplacian, sobelx, sobely
    
    def image_thresholding(self, img):
        """Apply different types of thresholding"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Simple thresholding
        ret, thresh_binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        ret, thresh_binary_inv = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
        ret, thresh_trunc = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)
        ret, thresh_tozero = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO)
        
        # Adaptive thresholding
        adaptive_mean = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                           cv.THRESH_BINARY, 11, 2)
        adaptive_gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv.THRESH_BINARY, 11, 2)
        
        # Otsu's thresholding
        ret, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        return (thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero,
                adaptive_mean, adaptive_gaussian, otsu)
    
    def morphological_operations(self, img):
        """Apply morphological transformations"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        
        # Define kernel
        kernel = np.ones((5, 5), np.uint8)
        
        # Morphological operations
        erosion = cv.erode(binary, kernel, iterations=1)
        dilation = cv.dilate(binary, kernel, iterations=1)
        opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
        gradient = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)
        tophat = cv.morphologyEx(binary, cv.MORPH_TOPHAT, kernel)
        blackhat = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)
        
        return erosion, dilation, opening, closing, gradient, tophat, blackhat
    
    def watershed_segmentation(self, img):
        """Apply watershed segmentation"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Apply threshold
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv.watershed(img, markers)
        img_watershed = img.copy()
        img_watershed[markers == -1] = [255, 0, 0]
        
        return img_watershed, sure_fg, sure_bg, unknown
    
    def grabcut_segmentation(self, img):
        """Apply GrabCut algorithm for foreground extraction"""
        height, width = img.shape[:2]
        
        # Create initial mask and models
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Define rectangle (adjust based on image content)
        rect = (width//4, height//4, width//2, height//2)
        
        # Apply GrabCut
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        
        # Create final mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = img * mask2[:, :, np.newaxis]
        
        return result, mask2
    
    def process_all_images(self):
        """Process all loaded images with all techniques"""
        for idx, img in enumerate(self.images):
            print(f"\nProcessing Image {idx + 1}...")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 25))
            gs = GridSpec(8, 4, figure=fig)
            
            # Original image
            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            ax.set_title('Original Image')
            ax.axis('off')
            
            # Canny Edge Detection
            edges = self.canny_edge_detection(img)
            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(edges, cmap='gray')
            ax.set_title('Canny Edges')
            ax.axis('off')
            
            # Hough Lines
            img_lines, img_lines_p = self.hough_line_detection(img)
            ax = fig.add_subplot(gs[0, 2])
            ax.imshow(cv.cvtColor(img_lines, cv.COLOR_BGR2RGB))
            ax.set_title('Hough Lines')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[0, 3])
            ax.imshow(cv.cvtColor(img_lines_p, cv.COLOR_BGR2RGB))
            ax.set_title('Probabilistic Hough Lines')
            ax.axis('off')
            
            # Hough Circles
            img_circles = self.hough_circle_detection(img)
            ax = fig.add_subplot(gs[1, 0])
            ax.imshow(cv.cvtColor(img_circles, cv.COLOR_BGR2RGB))
            ax.set_title('Hough Circles')
            ax.axis('off')
            
            # Color space conversions
            hsv, gray, mask_green, result_green = self.color_space_conversion_and_tracking(img)
            ax = fig.add_subplot(gs[1, 1])
            ax.imshow(cv.cvtColor(hsv, cv.COLOR_HSV2RGB))
            ax.set_title('HSV Color Space')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[1, 2])
            ax.imshow(gray, cmap='gray')
            ax.set_title('Grayscale')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[1, 3])
            ax.imshow(cv.cvtColor(result_green, cv.COLOR_BGR2RGB))
            ax.set_title('Green Object Tracking')
            ax.axis('off')
            
            # Histogram
            hist, hist_color = self.calculate_histogram(img)
            ax = fig.add_subplot(gs[2, 0])
            ax.plot(hist)
            ax.set_title('Grayscale Histogram')
            ax.set_xlim([0, 256])
            
            ax = fig.add_subplot(gs[2, 1])
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                ax.plot(hist_color[i], color=color)
            ax.set_title('Color Histogram')
            ax.set_xlim([0, 256])
            
            # Image gradients
            laplacian, sobelx, sobely = self.image_gradients(img)
            ax = fig.add_subplot(gs[2, 2])
            ax.imshow(laplacian, cmap='gray')
            ax.set_title('Laplacian')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[2, 3])
            ax.imshow(sobelx, cmap='gray')
            ax.set_title('Sobel X')
            ax.axis('off')
            
            # More gradients
            ax = fig.add_subplot(gs[3, 0])
            ax.imshow(sobely, cmap='gray')
            ax.set_title('Sobel Y')
            ax.axis('off')
            
            # Thresholding
            thresh_results = self.image_thresholding(img)
            thresh_names = ['Binary', 'Binary Inv', 'Truncated', 'To Zero',
                           'Adaptive Mean', 'Adaptive Gaussian', 'Otsu']
            
            for i, (thresh, name) in enumerate(zip(thresh_results[:3], thresh_names[:3])):
                ax = fig.add_subplot(gs[3, i+1])
                ax.imshow(thresh, cmap='gray')
                ax.set_title(name)
                ax.axis('off')
            
            # More thresholding
            for i, (thresh, name) in enumerate(zip(thresh_results[3:6], thresh_names[3:6])):
                ax = fig.add_subplot(gs[4, i])
                ax.imshow(thresh, cmap='gray')
                ax.set_title(name)
                ax.axis('off')
            
            ax = fig.add_subplot(gs[4, 3])
            ax.imshow(thresh_results[6], cmap='gray')
            ax.set_title('Otsu')
            ax.axis('off')
            
            # Morphological operations
            morph_results = self.morphological_operations(img)
            morph_names = ['Erosion', 'Dilation', 'Opening', 'Closing']
            
            for i, (morph, name) in enumerate(zip(morph_results[:4], morph_names)):
                ax = fig.add_subplot(gs[5, i])
                ax.imshow(morph, cmap='gray')
                ax.set_title(name)
                ax.axis('off')
            
            # More morphological operations
            morph_names_2 = ['Gradient', 'Top Hat', 'Black Hat']
            for i, (morph, name) in enumerate(zip(morph_results[4:7], morph_names_2)):
                ax = fig.add_subplot(gs[6, i])
                ax.imshow(morph, cmap='gray')
                ax.set_title(name)
                ax.axis('off')
            
            # Watershed segmentation
            watershed_result, sure_fg, sure_bg, unknown = self.watershed_segmentation(img)
            ax = fig.add_subplot(gs[6, 3])
            ax.imshow(cv.cvtColor(watershed_result, cv.COLOR_BGR2RGB))
            ax.set_title('Watershed Segmentation')
            ax.axis('off')
            
            # GrabCut segmentation
            grabcut_result, grabcut_mask = self.grabcut_segmentation(img)
            ax = fig.add_subplot(gs[7, 0])
            ax.imshow(cv.cvtColor(grabcut_result, cv.COLOR_BGR2RGB))
            ax.set_title('GrabCut Foreground')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[7, 1])
            ax.imshow(grabcut_mask, cmap='gray')
            ax.set_title('GrabCut Mask')
            ax.axis('off')
            
            # Additional watershed results
            ax = fig.add_subplot(gs[7, 2])
            ax.imshow(sure_fg, cmap='gray')
            ax.set_title('Watershed - Sure Foreground')
            ax.axis('off')
            
            ax = fig.add_subplot(gs[7, 3])
            ax.imshow(unknown, cmap='gray')
            ax.set_title('Watershed - Unknown Region')
            ax.axis('off')
            
            plt.tight_layout()
            plt.suptitle(f'Comprehensive OpenCV Analysis - Image {idx + 1}', y=0.98)
            plt.show()

# Usage example - you need to provide the actual paths to your images
def main():
    # Replace these with the actual paths to your uploaded images
    image_paths = [
        'image1.jpg',  # Replace with actual path to first image (squirrel)
        'image2.jpg',  # Replace with actual path to second image (park)
        'image3.jpg'   # Replace with actual path to third image (campus)
    ]
    
    # Alternative: if images are in current directory with these names
    # image_paths = ['squirrel.jpg', 'park.jpg', 'campus.jpg']
    
    processor = ComprehensiveImageProcessor(image_paths)
    
    if processor.images:
        processor.process_all_images()
        print("\nProcessing complete! All OpenCV techniques have been applied.")
        print("\nTechniques demonstrated:")
        print("1. Canny Edge Detection")
        print("2. Hough Line Transform (Standard and Probabilistic)")
        print("3. Hough Circle Transform")
        print("4. Color Space Conversion (BGR→HSV, BGR→Gray)")
        print("5. Object Tracking (Color-based)")
        print("6. Histogram Calculation and Analysis")
        print("7. Image Gradients (Laplacian, Sobel X&Y)")
        print("8. Image Thresholding (Simple, Adaptive, Otsu)")
        print("9. Morphological Transformations (Erosion, Dilation, Opening, Closing, etc.)")
        print("10. Watershed Segmentation")
        print("11. GrabCut Foreground Extraction")
    else:
        print("No images were loaded successfully. Please check the file paths.")
        print("Make sure to update the image_paths list with the correct paths to your images.")

if __name__ == "__main__":
    main()
