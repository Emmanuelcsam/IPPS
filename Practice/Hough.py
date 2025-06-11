import numpy as np
import cv2 as cv
import os

base_path = 'C:/Users/Saem1001/Documents/GitHub/OpenCV-Practice/'
image_path = base_path + '19700103045135-J67690-FT41.jpg'
 
if not os.path.exists(image_path):
    print(f"Error: File could not be read. Check if '{image_path}' exists.")
else:
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image at path '{image_path}' could not be loaded. It might be corrupted or in an unsupported format.")
    else:
        img_blur = cv.medianBlur(img, 5)
        cimg = cv.cvtColor(img_blur, cv.COLOR_GRAY2BGR)
 
        circles = cv.HoughCircles(
            image=img_blur,                  
            method=cv.HOUGH_GRADIENT,        # Detection method
            dp=1,                            # Inverse ratio of accumulator resolution. 1 = same resolution.
            minDist=20,                      # Minimum distance between centers of detected circles.
            param1=50,                       # Upper threshold for the internal Canny edge detector.
            param2=30,                       # Accumulator threshold for circle centers. A lower value detects more circles (and more false positives).
            minRadius=0,                     # Minimum radius. Setting to 0 detects circles of any size.
            maxRadius=0                      # Maximum radius. Setting to 0 detects circles of any size.
        )
        # Check if circles were detected
        if circles is not None:
            print(f"Success! Detected {len(circles[0])} circle(s).")
            # Convert the circle parameters (x, y, radius) to integers
            circles = np.uint16(np.around(circles))
 
            # Loop through all detected circles
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                # Draw the outer circle outline in bright green
                cv.circle(cimg, center, radius, (0, 255, 0), 2)
                # Draw a small red dot at the center of the circle
                cv.circle(cimg, center, 2, (0, 0, 255), 3)
 
            # Display the final image with the detected circles
            cv.imshow('Detected Circles', cimg)
            print("Press any key while the image window is active to close it.")
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("No circles were detected in the image with the current parameters.")