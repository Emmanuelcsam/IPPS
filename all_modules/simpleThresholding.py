import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

base_path = 'C:/Users/Saem1001/Documents/GitHub/OpenCV-Practice/'
img_path = base_path + 'img1.jpg' 
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
# Apply binary thresholding: pixels > 127 become 255, others 0
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Apply inverse binary thresholding: pixels > 127 become 0, others 255
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# Apply truncated thresholding: pixels > 127 become 127, others unchanged
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
# Apply threshold to zero: pixels > 127 unchanged, others 0
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
# Apply inverse threshold to zero: pixels > 127 become 0, others unchanged
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
 
# Define titles for each of the images to be displayed
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# Create a list of images: original and all thresholded versions
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
 
# Loop through the images and titles to display them
for i in range(6):
    # Create a subplot in a 2x3 grid at position i+1
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    # Set the title for the current subplot
    plt.title(titles[i])
    # Remove x-axis ticks and labels
    plt.xticks([])
    # Remove y-axis ticks and labels
    plt.yticks([])
 
# Display all the created subplots
plt.show()