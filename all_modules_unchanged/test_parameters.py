import cv2

# Load and preprocess image
img_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Test different parameter combinations
test_params = [
    (1.0, 50, 30),
    (1.2, 80, 25),
    (1.5, 100, 20)
]

for dp, param1, param2 in test_params:
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=30,
        param1=param1,
        param2=param2,
        minRadius=5,
        maxRadius=200
    )
    
    n_circles = 0 if circles is None else len(circles[0])
    print(f"dp={dp}, param1={param1}, param2={param2}: {n_circles} circles found")