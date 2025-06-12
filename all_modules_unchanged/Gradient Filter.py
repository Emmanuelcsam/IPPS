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

    # Calculate the gradient magnitude
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Convert gradient magnitude to absolute 8-bit integers for display
    gradient_image = cv2.convertScaleAbs(gradient_magnitude)

    # Display the original image and its gradient magnitude
    cv2.imshow('Original Image', image)
    cv2.imshow('Gradient Magnitude', gradient_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()