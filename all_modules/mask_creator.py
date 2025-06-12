import cv2
import numpy as np

def create_circular_mask(shape, center, radius):
    """Create circular mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, (255,), -1)
    return mask

def create_annular_mask(shape, center, inner_r, outer_r):
    """Create ring-shaped mask."""
    outer_mask = create_circular_mask(shape, center, outer_r)
    inner_mask = create_circular_mask(shape, center, inner_r)
    return cv2.subtract(outer_mask, inner_mask)

def apply_mask(image, mask):
    """Apply mask to image."""
    return cv2.bitwise_and(image, image, mask=mask)

if __name__ == "__main__":

    img = cv2.imread("fiber_optic_image.jpg", cv2.IMREAD_GRAYSCALE)
    
    # isolate core (assuming center at image center)
    h, w = img.shape
    center = (w//2, h//2)
    
    # Core mask
    core_mask = create_circular_mask(img.shape, center, 50)
    core_region = apply_mask(img, core_mask)
    cv2.imwrite("core_region.jpg", core_region)
    
    # Cladding mask
    clad_mask = create_annular_mask(img.shape, center, 50, 100)
    clad_region = apply_mask(img, clad_mask)
    cv2.imwrite("cladding_region.jpg", clad_region)