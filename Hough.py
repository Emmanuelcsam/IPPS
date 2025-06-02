import numpy as np

import cv2 as cv

import os
 
image_path = 'image2.jpg'

assert os.path.exists(image_path), f"File could not be read. Check if '{image_path}' exists."

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE) #reads img in grayscale

img = cv.medianBlur(img, 5) # blur to reduce noise (5x5 kernel)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) #creates a colored version of image for drawing circles
# Use the Hough Circle Transform algorithm to find circles.
#Hough transform
circles = cv.HoughCircles(img,                     # Source image

                          cv.HOUGH_GRADIENT,       # Detection method

                          dp=1,                    # Inverse ratio of accumulator resolution

                          minDist=100,             # Minimum distance between centers of detected circles

                          param1=100,              # Upper threshold for the Canny edge detector

                          param2=30,               # Accumulator threshold for circle centers (lower means more circles)

                          minRadius=100,           # Minimum radius of a circle to be detected

                          maxRadius=300)           # Maximum radius of a circle to be detected
 
 
# --- 4. Draw Detected Circles ---

# Ensure that at least one circle was found

if circles is not None:

    # Convert the circle parameters (x, y, radius) to integers

    circles = np.uint16(np.around(circles))
 
    # Loop through all detected circles

    for i in circles[0, :]:

        center = (i[0], i[1])

        radius = i[2]

        # Draw the outer boundary of the circle in green

        cv.circle(cimg, center, radius, (0, 255, 0), 2)

        # Draw the center of the circle in red

        cv.circle(cimg, center, 2, (0, 0, 255), 3)
 
    # --- 5. Display the Result ---

    cv.imshow('Detected Circles', cimg)

    cv.waitKey(0)  # Wait for a key press to close the image window

    cv.destroyAllWindows()

else:

    print("No circles were detected in the image.")
 