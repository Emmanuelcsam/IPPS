import cv2
import numpy as np
import argparse

class FiberOpticAnalyzer:
    """
    A comprehensive tool for analyzing fiber optic end face images.

    This class provides methods to:
    - Detect the core, cladding, and ferrule.
    - Calculate their pixel distances (diameters) and areas.
    - Detect defects such as scratches and digs.
    - Label the components and defects on the image.
    """

    def __init__(self, image_path):
        """
        Initializes the analyzer with the path to the fiber optic image.
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.output_image = self.image.copy()
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def detect_components(self):
        """
        Detects the core, cladding, and ferrule using the Hough Circle Transform.
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_image, (9, 9), 2)

        # Use Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=200
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Sort circles by radius to identify core, cladding, and ferrule
            sorted_circles = sorted(circles[0, :], key=lambda x: x[2])

            if len(sorted_circles) >= 3:
                self.core = sorted_circles[0]
                self.cladding = sorted_circles[1]
                self.ferrule = sorted_circles[2]
                return True
        return False

    def calculate_measurements(self):
        """
        Calculates the pixel distance (diameter) and area of the detected components.
        """
        if hasattr(self, 'core'):
            core_diameter = self.core[2] * 2
            core_area = np.pi * (self.core[2] ** 2)
            print(f"Core: Diameter = {core_diameter}px, Area = {core_area:.2f}px^2")

            cladding_diameter = self.cladding[2] * 2
            cladding_area = np.pi * (self.cladding[2] ** 2)
            print(f"Cladding: Diameter = {cladding_diameter}px, Area = {cladding_area:.2f}px^2")

            ferrule_diameter = self.ferrule[2] * 2
            ferrule_area = np.pi * (self.ferrule[2] ** 2)
            print(f"Ferrule: Diameter = {ferrule_diameter}px, Area = {ferrule_area:.2f}px^2")

    def detect_defects(self):
        """
        Detects scratches and digs on the fiber optic end face.
        """
        # Scratch detection using Hough Line Transform
        edges = cv2.Canny(self.gray_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is not None:
            print(f"Detected {len(lines)} potential scratches.")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(self.output_image, 'Scratch', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Dig detection using adaptive thresholding and contour detection
        thresh = cv2.adaptiveThreshold(self.gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dig_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 500:  # Filter for small defects
                dig_count += 1
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(self.output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(self.output_image, 'Dig', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if dig_count > 0:
            print(f"Detected {dig_count} potential digs.")

    def label_image(self):
        """
        Labels the detected components on the output image.
        """
        if hasattr(self, 'core'):
            # Label Core
            cv2.circle(self.output_image, (self.core[0], self.core[1]), self.core[2], (0, 255, 0), 2)
            cv2.putText(self.output_image, 'Core', (self.core[0] - 20, self.core[1] - self.core[2] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Label Cladding
            cv2.circle(self.output_image, (self.cladding[0], self.cladding[1]), self.cladding[2], (0, 255, 255), 2)
            cv2.putText(self.output_image, 'Cladding', (self.cladding[0] - 40, self.cladding[1] - self.cladding[2] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Label Ferrule
            cv2.circle(self.output_image, (self.ferrule[0], self.ferrule[1]), self.ferrule[2], (255, 0, 255), 2)
            cv2.putText(self.output_image, 'Ferrule', (self.ferrule[0] - 40, self.ferrule[1] - self.ferrule[2] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    def show_results(self):
        """
        Displays the original and analyzed images.
        """
        cv2.imshow('Original Image', self.image)
        cv2.imshow('Analyzed Image', self.output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Analyze a fiber optic end face image.')
    parser.add_argument('image_path', type=str, help='The path to the image file.')
    args = parser.parse_args()

    try:
        analyzer = FiberOpticAnalyzer(args.image_path)
        if analyzer.detect_components():
            analyzer.calculate_measurements()
            analyzer.label_image()
        else:
            print("Could not detect the core, cladding, and ferrule.")

        analyzer.detect_defects()
        analyzer.show_results()

    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()

