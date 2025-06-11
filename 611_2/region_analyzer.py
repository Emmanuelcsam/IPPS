import cv2
import numpy as np

def analyze_region(masked_image):
    """Get statistics from masked region."""
    pixels = masked_image[masked_image > 0]
    
    if len(pixels) == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
    
    return {
        'mean': np.mean(pixels),
        'std': np.std(pixels),
        'min': np.min(pixels),
        'max': np.max(pixels),
        'count': len(pixels)
    }

def quick_stats(image_path, mask_path=None):
    """Quick statistics from image or masked region."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.bitwise_and(img, img, mask=mask)
    
    stats = analyze_region(img)
    print(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    print(f"Range: [{stats['min']}, {stats['max']}], Pixels: {stats['count']}")
    return stats

if __name__ == "__main__":

    stats = quick_stats(r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg")
    
