import cv2
import numpy as np

def create_circle_mask(shape, center, radius):
    """Create a circular mask"""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.circle(mask, center, radius, (255,), -1)
    return mask

def split_circle(img, inner_circle, outer_circle):
    """Split circle image into inner circle and outer ring"""
    if inner_circle is None or outer_circle is None:
        return None, None
    
    # Create masks
    inner_mask = create_circle_mask(img.shape, tuple(inner_circle[:2]), inner_circle[2])
    outer_mask = create_circle_mask(img.shape, tuple(outer_circle[:2]), outer_circle[2])
    ring_mask = cv2.subtract(outer_mask, inner_mask)
    
    # Apply masks
    inner_img = cv2.bitwise_and(img, img, mask=inner_mask)
    ring_img = cv2.bitwise_and(img, img, mask=ring_mask)
    
    return inner_img, ring_img

def crop_to_content(img, mask):
    """Crop image to non-zero content based on mask"""
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y+h, x:x+w]
    return img

if __name__ == "__main__":
    import sys
    from circle_detector import inner_outer_split
    
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        inner, outer = inner_outer_split(img)
        
        if inner is not None and outer is not None:
            inner_img, ring_img = split_circle(img, inner, outer)
            
            #crop
            inner_mask = create_circle_mask(img.shape, tuple(inner[:2]), inner[2])
            outer_mask = create_circle_mask(img.shape, tuple(outer[:2]), outer[2])
            ring_mask = cv2.subtract(outer_mask, inner_mask)
            
            inner_cropped = crop_to_content(inner_img, inner_mask)
            ring_cropped = crop_to_content(ring_img, ring_mask)
            
            if inner_cropped is not None:
                cv2.imwrite("inner.png", inner_cropped)
            if ring_cropped is not None:
                cv2.imwrite("ring.png", ring_cropped)
            print("Saved inner.png and ring.png")