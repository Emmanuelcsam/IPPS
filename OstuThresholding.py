#Otsu Method
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

base_path = 'C:/Users/Saem1001/Documents/GitHub/OpenCV-Practice/'
img_path = base_path + 'img1.jpg' 
img = cv.imread(base_path + 'img1.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Case 1: Global thresholding with a fixed value of 127. [cite: 27, 30]
ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Case 2: Otsu's thresholding. [cite: 28]
# The threshold value is set to 0, but it will be ignored because cv.THRESH_OTSU is used.
# The function automatically calculates the optimal threshold (ret2). [cite: 25]
ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Case 3: Gaussian filtering followed by Otsu's thresholding.
# Filtering removes noise, which helps create a better histogram for Otsu's method. [cite: 28, 29]
blur = cv.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Plotting the images and their corresponding histograms to show the effect
# of each method. [cite: 29]
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

# The loop structure is more complex to accommodate the histograms in the plot.
for i in range(3):
    # Plot the source image (original or blurred)
    plt.subplot(3, 3, i * 3 + 1)
    plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    
    # Plot the histogram of the source image
    plt.subplot(3, 3, i * 3 + 2)
    plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])

    # Plot the thresholded result
    plt.subplot(3, 3, i * 3 + 3)
    plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])

plt.show()

