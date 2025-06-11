import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_image(image_path):
    """Load image from file path"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image

def plot_histogram(image_path, mask_path=None, title="Intensity Histogram"):
    """Plot histogram of image intensities."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pixels = img[mask > 0]
    else:
        pixels = img.flatten()
    
    plt.figure(figsize=(8, 6))
    plt.hist(pixels[pixels > 0], bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_comparison(original_path, processed_path, title1="Original", title2="Processed"):
    """Side-by-side comparison."""
    img1 = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def quick_view(image_paths, titles=None, cmap='gray'):
    """Quick view of multiple images."""
    n = len(image_paths)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(4*cols, 4*rows))
    for i, path in enumerate(image_paths):
        plt.subplot(rows, cols, i+1)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap=cmap)
        plt.title(titles[i] if titles else f"Image {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
    try:
        image = load_image(image_path)
        print(f"Successfully loaded image from {image_path} with shape {image.shape}")
        
        cv2.imshow("Loaded Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")    

    plot_histogram("fiber_optic_image.jpg", title="Fiber Optic Intensity")
    visualize_comparison("fiber_optic_image.jpg", "defects_edges.jpg", "Original", "Defects")
    
    # View multiple results
    quick_view(["fiber_optic_image.jpg", "core_region.jpg", "defects_edges.jpg"],
               ["Original", "Core Region", "Defects"])

