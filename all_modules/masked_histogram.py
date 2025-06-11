import cv2
import numpy as np

# Calculate histogram with mask
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    # Create a circular mask (example)
    h, w = img.shape
    center = (w // 2, h // 2)
    radius = min(h, w) // 3
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Calculate histogram with mask
    hist = cv2.calcHist([img], [0], mask, [256], [0, 256])
    
    # Create histogram visualization
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / 256))
    
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    
    # Normalize histogram
    cv2.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    
    # Draw histogram
    for i in range(1, 256):
        cv2.line(histImage,
                (bin_w * (i-1), hist_h - int(hist[i-1])),
                (bin_w * i, hist_h - int(hist[i])),
                (255, 255, 255), 2)
    
    # Show mask, masked image, and histogram
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('Mask', mask)
    cv2.imshow('Masked Image', masked_img)
    cv2.imshow('Masked Histogram', histImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Failed to load image from {image_path}")