import cv2

# Load and preprocess image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Target number of circles to find
target_circles = 2

# Try different parameters
found = False
for param1 in [30, 50, 80, 100]:
    for param2 in [15, 20, 25, 30, 35]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30, param1=param1, param2=param2,
            minRadius=5, maxRadius=200
        )
        
        if circles is not None and len(circles[0]) == target_circles:
            print(f"Found {target_circles} circles!")
            print(f"Parameters: param1={param1}, param2={param2}")
            found = True
            break
    if found:
        break

if not found:
    print(f"Could not find exactly {target_circles} circles")