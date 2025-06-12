import cv2
import numpy as np

# Load the image
image = cv2.imread(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
else:
    # Perform 2D Fourier Transform
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Create a high-pass filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0

    # Apply mask
    fshift = dft_shift * mask

    # Perform inverse Fourier Transform
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Display the original and Fourier-based high-pass filtered images
    cv2.imshow('Original Image', image)
    cv2.imshow('Fourier High-Pass Filtered', np.uint8(img_back))
    cv2.waitKey(0)
    cv2.destroyAllWindows()