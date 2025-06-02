import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_and_display_image(image_path, image_name, is_squirrel_image=False, is_steps_image=False):
    """
    Loads an image, applies various OpenCV processing techniques,
    and displays the results using Matplotlib.
    """
    # Load the original image
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to Grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 1. Gaussian Blurring
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 2. Canny Edge Detection
    edges = cv2.Canny(img_blur, 50, 150) # Lowered thresholds for potentially more detail

    # 3. Otsu's Thresholding
    # Gaussian blur is often recommended before Otsu's
    blur_for_otsu = cv2.GaussianBlur(img_gray, (5,5),0)
    _, otsu_thresh = cv2.threshold(blur_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Adaptive Thresholding (Gaussian)
    adaptive_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                           cv2.THRESH_BINARY, 11, 2)

    # 5. Sobel Gradients
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)
    sobel_x_abs = np.absolute(sobel_x)
    sobel_y_abs = np.absolute(sobel_y)
    sobel_x_8u = np.uint8(sobel_x_abs)
    sobel_y_8u = np.uint8(sobel_y_abs)

    # 6. Morphological Operations (on Otsu's thresholded image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # Kernel for morphology
    morph_open = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel)
    morph_close = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)

    # 7. Hough Line Transform (especially for steps_image)
    lines_img = img_color.copy()
    detected_lines = None
    if is_steps_image:
        # Use Canny edges as input for Hough Lines
        canny_for_lines = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(canny_for_lines, 1, np.pi / 180, threshold=80, # Adjusted threshold
                                minLineLength=50, maxLineGap=10)
        if lines is not None:
            detected_lines = True
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            print(f"No lines detected for {image_name}")
            detected_lines = False


    # 8. Hough Circle Transform (attempt)
    circles_img = img_color.copy()
    # Parameters for HoughCircles can be tricky and image-dependent
    # Using parameters similar to the PDF example, may need tuning
    # It works best on grayscale images
    img_gray_blur_for_circles = cv2.medianBlur(img_gray, 5) # Median blur can be good for circle detection
    detected_circles = None
    circles = cv2.HoughCircles(img_gray_blur_for_circles, cv2.HOUGH_GRADIENT, dp=1, minDist=40, # Increased minDist
                               param1=50, param2=30, minRadius=5, maxRadius=50) # Adjusted radius
    if circles is not None:
        detected_circles = True
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(circles_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(circles_img, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        print(f"No circles detected for {image_name}")
        detected_circles = False

    # Plotting the results
    plt.figure(figsize=(20, 18)) # Adjusted figure size
    plt.suptitle(f"Image Processing for: {image_name}", fontsize=16)

    titles = ['Original Color', 'Grayscale', 'Gaussian Blur', 'Canny Edges',
              'Otsu Threshold', 'Adaptive Threshold', 'Sobel X', 'Sobel Y',
              'Morphological Opening', 'Morphological Closing']
    images = [cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB), img_gray, img_blur, edges,
              otsu_thresh, adaptive_thresh, sobel_x_8u, sobel_y_8u,
              morph_open, morph_close]

    for i in range(10):
        plt.subplot(4, 3, i + 1)
        plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    # Plot Histogram
    plt.subplot(4, 3, 10)
    plt.hist(img_gray.ravel(), 256, [0, 256])
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')

    # Plot Hough Lines result
    plt.subplot(4, 3, 11)
    if is_steps_image and detected_lines:
        plt.imshow(cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB))
        plt.title('Hough Lines')
    elif is_steps_image and not detected_lines:
         # Show a blank or original if no lines
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.title('Hough Lines (None Detected)')
    else: # For squirrel image or if not applicable
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)) # Placeholder
        plt.title('Hough Lines (Not Run)')
    plt.xticks([]), plt.yticks([])


    # Plot Hough Circles result
    plt.subplot(4, 3, 12)
    if detected_circles:
        plt.imshow(cv2.cvtColor(circles_img, cv2.COLOR_BGR2RGB))
        plt.title('Hough Circles')
    elif not detected_circles:
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.title('Hough Circles (None Detected)')
    plt.xticks([]), plt.yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    # Define image paths (make sure these files are in the same directory as the script)
    image_file_squirrel = '184A2174.jpg'
    image_file_steps = '184A2261.jpg'

    print(f"Processing {image_file_squirrel}...")
    process_and_display_image(image_file_squirrel, "Squirrel in Grass", is_squirrel_image=True)

    print(f"\nProcessing {image_file_steps}...")
    process_and_display_image(image_file_steps, "People on Steps", is_steps_image=True)

    print("\nAll processing complete.")

