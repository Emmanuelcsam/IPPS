import numpy as np
import cv2 as cv
 
def circle_extract(image: np.ndarray, x0: np.int16, y0: np.int16, radius: np.float16) -> np.ndarray:
    """Takes in a grayscale image"""
    arr = image
    rows, cols = arr.shape
    #print(arr.shape)
    #print("rows =", rows)
    #print("cols =", cols)
 
    for i in range(rows):
        for j in range(cols):
            if np.sqrt(np.square(i - y0) + np.square(j - x0)) > radius:
                arr[i][j] = 0
            else:
                pass
    cv.imshow("cropped image", arr)
    return arr
 
 
def main():
    img = cv.imread('C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg', cv.IMREAD_GRAYSCALE)
   
    assert img is not None, "file could not be read, check with os.path.exists()"
   
    img = cv.medianBlur(img,5)
   
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    # Detects Core
    #circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT,dp=1, minDist=10,
    #                      param1=50, param2=40, minRadius=20, maxRadius=50)
    # Detects Cladding
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT,dp=1, minDist=10,
                             param1=50, param2=40, minRadius=100, maxRadius=500)
    print(type(circles))
    print("circles =",circles)
    circles2 = np.uint16(np.around(circles))
    print("circles2 =", circles)
    print(circles2.shape)
    x0 = circles[0][0][0]
    y0 = circles[0][0][1]
    radius = circles[0][0][2]
 
 
    for i in circles2[0,:]:
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #print(type(img))
    circle_extract(img, x0, y0, radius)
 
    cv.imshow('detected circles',cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
