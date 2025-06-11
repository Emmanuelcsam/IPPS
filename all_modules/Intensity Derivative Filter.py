import cv2
import numpy as np

# Load the image
image = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    # Calculate Sobel derivatives for X and Y axes
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Convert derivatives to absolute 8-bit integers for display
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # Display the original image and its intensity derivatives
    cv2.imshow('Original Image', image)
    cv2.imshow('Sobel X Derivative', abs_sobel_x)
    cv2.imshow('Sobel Y Derivative', abs_sobel_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()