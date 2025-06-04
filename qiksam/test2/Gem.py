import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

def process_and_display_image(image_path):
    """
    Loads a fiber optic image, analyzes it for components and defects,
    and displays the results.
    """
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"Error: Could not load image at {image_path}")
        return

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    output_image = img_color.copy()

    # --- Component Detection ---
    blurred = cv2.GaussianBlur(img_gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=10, maxRadius=200
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        sorted_circles = sorted(circles[0, :], key=lambda x: x[2])

        if len(sorted_circles) >= 3:
            core, cladding, ferrule = sorted_circles[0], sorted_circles[1], sorted_circles[2]

            # --- Measurements ---
            print(f"Core Diameter: {core[2] * 2}px, Cladding Diameter: {cladding[2] * 2}px, Ferrule Diameter: {ferrule[2] * 2}px")

            # --- Labeling ---
            cv2.circle(output_image, (core[0], core[1]), core[2], (0, 255, 0), 2)
            cv2.putText(output_image, 'Core', (core[0] - 20, core[1] - core[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.circle(output_image, (cladding[0], cladding[1]), cladding[2], (0, 255, 255), 2)
            cv2.putText(output_image, 'Cladding', (cladding[0] - 40, cladding[1] - cladding[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.circle(output_image, (ferrule[0], ferrule[1]), ferrule[2], (255, 0, 255), 2)
            cv2.putText(output_image, 'Ferrule', (ferrule[0] - 40, ferrule[1] - ferrule[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    else:
        print("Could not detect fiber components.")

    # --- Defect Detection (Simplified) ---
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- Display Results ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Analyzed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='A simplified tool for fiber optic image analysis.')
    parser.add_argument('image_path', type=str, help='The path to the image file.')
    args = parser.parse_args()

    process_and_display_image(args.image_path)

if __name__ == "__main__":
    main()
