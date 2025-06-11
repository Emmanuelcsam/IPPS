import cv2
import numpy as np

def auto_tune_circles(image_path, target_circles=2):
    """Auto-tune parameters to find target number of circles."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Parameter ranges to try
    dp_values = [1.0, 1.2, 1.5]
    param1_values = [30, 50, 80, 100]
    param2_values = [15, 20, 25, 30, 35]
    
    best_params = None
    best_circles = None
    
    for dp in dp_values:
        for p1 in param1_values:
            for p2 in param2_values:
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT,
                    dp=dp, minDist=30, param1=p1, param2=p2,
                    minRadius=5, maxRadius=200
                )
                
                if circles is not None:
                    n_circles = len(circles[0])
                    if n_circles == target_circles:
                        print(f"Found {n_circles} circles with dp={dp}, param1={p1}, param2={p2}")
                        return circles[0]
                    
                    if best_circles is None or abs(n_circles - target_circles) < abs(len(best_circles) - target_circles):
                        best_circles = circles[0]
                        best_params = (dp, p1, p2)
    
    if best_params is not None:
        dp, p1, p2 = best_params
        assert best_circles is not None
        print(f"Best match: {len(best_circles)} circles with dp={dp}, param1={p1}, param2={p2}")
        return best_circles
    
    print("No circles found")
    return None

def interactive_tune(image_path):
    """Simple interactive parameter tuning."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def detect_and_draw(dp, param1, param2):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=dp/10, minDist=50, param1=param1, param2=param2,
            minRadius=10, maxRadius=150
        )
        
        result = img.copy()
        if circles is not None:
            circles = np.around(circles[0]).astype(np.uint16)
            for x, y, r in circles:
                cv2.circle(result, (x, y), r, (0, 255, 0), 2)
                cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
        
        cv2.imshow('Result', result)
    
    # Create trackbars
    cv2.namedWindow('Result')
    cv2.createTrackbar('dp*10', 'Result', 12, 30, lambda x: None)
    cv2.createTrackbar('param1', 'Result', 50, 200, lambda x: None)
    cv2.createTrackbar('param2', 'Result', 30, 100, lambda x: None)
    
    print("Adjust trackbars to find circles. Press 'q' to quit.")
    
    while True:
        dp = cv2.getTrackbarPos('dp*10', 'Result')
        param1 = cv2.getTrackbarPos('param1', 'Result')
        param2 = cv2.getTrackbarPos('param2', 'Result')
        
        detect_and_draw(dp, param1, param2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"Final parameters: dp={dp/10}, param1={param1}, param2={param2}")

if __name__ == "__main__":

    circles = auto_tune_circles(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg", target_circles=2)
    
