#Adaptive Thresholding
# Import necessary libraries. [cite: 18]
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

base_path = 'C:/Users/Saem1001/Documents/GitHub/OpenCV-Practice/'
img_path = base_path + 'img1.jpg' 
img = cv.imread(base_path + 'img1.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

#Applies blur
img = cv.medianBlur(img, 5)

# Perform global thresholding as a comparison. 
ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Perform adaptive thresholding using the mean method. [cite: 18]
# img: source image
# 255: maximum value
# cv.ADAPTIVE_THRESH_MEAN_C: adaptive method
# cv.THRESH_BINARY: threshold type
# 11: blockSize (neighborhood size)
# 2: C (the constant to subtract)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                           cv.THRESH_BINARY, 11, 2)

# Perform adaptive thresholding using the Gaussian method. [cite: 18]
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                           cv.THRESH_BINARY, 11, 2)

# Create lists of titles and images for plotting. [cite: 18]
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

# Loop to display the results in a 2x2 grid. [cite: 18]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
