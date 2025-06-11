import cv2
import numpy as np

def draw_circles(img, inner_circle, outer_circle):
    """Draw detected circles on image"""
    vis = img.copy()
    if inner_circle is not None:
        cv2.circle(vis, tuple(inner_circle[:2]), inner_circle[2], (0, 255, 0), 2)
    if outer_circle is not None:
        cv2.circle(vis, tuple(outer_circle[:2]), outer_circle[2], (0, 0, 255), 2)
    return vis

def resize_for_display(img, max_width=800):
    """Resize image if too large for display"""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def display_results(original, inner_img, ring_img, inner_circle, outer_circle):
    """Display all results in windows"""
    vis = draw_circles(original, inner_circle, outer_circle)
    
    # Resize for display
    vis = resize_for_display(vis)
    inner_display = resize_for_display(inner_img) if inner_img is not None else None
    ring_display = resize_for_display(ring_img) if ring_img is not None else None
    
    cv2.imshow('Detected Circles', vis)
    if inner_display is not None:
        cv2.imshow('Inner Circle', inner_display)
    if ring_display is not None:
        cv2.imshow('Outer Ring', ring_display)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_side_by_side(images, labels=None):
    """Create side-by-side comparison image"""
    # Ensure all images have same height
    max_h = max(img.shape[0] for img in images)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h < max_h:
            pad = (max_h - h) // 2
            img = cv2.copyMakeBorder(img, pad, max_h-h-pad, 0, 0, cv2.BORDER_CONSTANT)
        resized.append(img)
    
    # Concatenate horizontally
    result = np.hstack(resized)
    
    # Add labels if provided
    if labels:
        for i, label in enumerate(labels):
            x = sum(img.shape[1] for img in resized[:i]) + 10
            cv2.putText(result, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result

# Standalone usage example
if __name__ == "__main__":
    import sys
    from circle_detector import detect_washer_circles
    from washer_splitter import split_washer
    
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        inner, outer = detect_washer_circles(img)
        
        if inner is not None:
            inner_img, ring_img = split_washer(img, inner, outer)
            display_results(img, inner_img, ring_img, inner, outer)