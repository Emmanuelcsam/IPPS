import cv2
import numpy as np

# Load image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
noise = np.random.normal(0, 5, image.shape)
noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

# Display result
cv2.imshow('Original', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite('noise_result.jpg', noisy_image)