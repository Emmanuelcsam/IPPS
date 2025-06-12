import cv2
import numpy as np

# Load the image
image = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    # Convert to grayscale for filtering
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a high-pass filter kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

    # Apply the high-pass filter using convolution
    high_pass_image = cv2.filter2D(gray_image, -1, kernel)

    # Display the original and high-pass filtered images
    cv2.imshow('Original Image', image)
    cv2.imshow('High-Pass Filtered', high_pass_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()