import cv2
import numpy as np

# Display multiple images in a grid
image_paths = [
    r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg",
    r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg",  # Change to other paths
    r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
]

images = []
for path in image_paths:
    img = cv2.imread(path)
    if img is not None:
        images.append(img)

if images:
    # Resize all images to same size
    target_height = 300
    resized_images = []
    
    for img in images:
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_height))
        resized_images.append(resized)
    
    # Create grid (3 columns max)
    n = len(resized_images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    # Combine images
    row_images = []
    for i in range(0, n, cols):
        row = resized_images[i:i+cols]
        if len(row) < cols:
            # Pad with black images
            h, w, c = row[0].shape
            for _ in range(cols - len(row)):
                row.append(np.zeros((h, w, c), dtype=np.uint8))
        row_images.append(np.hstack(row))
    
    grid = np.vstack(row_images) if row_images else None
    
    if grid is not None:
        cv2.imshow('Grid View', grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Failed to load images")