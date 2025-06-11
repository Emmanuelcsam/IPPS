import cv2
import numpy as np

def detect_circles_hough(img):
    """Detect circles using Hough Transform"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (9, 9), 2),
        cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30,
        minRadius=10, maxRadius=min(img.shape[:2]) // 2
    )
    return circles[0] if circles is not None else None

def detect_circles_contours(img):
    """Detect circles using contours"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            (x, y), r = cv2.minEnclosingCircle(c)
            perimeter = cv2.arcLength(c, True)
            if 4 * np.pi * area / (perimeter ** 2) > 0.7:  # roundness check
                circles.append([int(x), int(y), int(r)])
    
    return sorted(circles, key=lambda x: x[2])[:2] if len(circles) >= 2 else None

def inner_outer_split(img):
    """detect inner and outer circles"""
    circles = detect_circles_hough(img)
    if circles is None or len(circles) < 2:
        circles = detect_circles_contours(img)
    
    if circles is not None and len(circles) >= 2:
        circles = sorted(circles, key=lambda x: x[2])  # Sort by radius
        return circles[0], circles[1]  # inner, outer
    return None, None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        inner, outer = inner_outer_split(img)
        if inner is not None:
            print(f"Inner: center=({inner[0]}, {inner[1]}), r={inner[2]}")
            print(f"Outer: center=({outer[0]}, {outer[1]}), r={outer[2]}")
        else:
            print("No circles detected")