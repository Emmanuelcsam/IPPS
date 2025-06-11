import cv2
import numpy as np

# Load image
image_path = r"C:\Users\Saem1001\Documents\GitHub\OpenCV-Practice\samples2\img38.jpg"
image = cv2.imread(image_path)

# Define circles (example values - adjust based on your detection)
inner_circle = (200, 200, 80)  # Adjust these values
outer_circle = (200, 200, 150)  # Adjust these values

# Create visualization
vis_image = image.copy()

# Draw circles
cv2.circle(vis_image, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 255, 0), 2)
cv2.circle(vis_image, (outer_circle[0], outer_circle[1]), outer_circle[2], (0, 0, 255), 2)

# Add labels
cv2.putText(vis_image, 'Inner', (inner_circle[0]-20, inner_circle[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(vis_image, 'Outer', (outer_circle[0]-20, outer_circle[1]+outer_circle[2]+20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Display result
cv2.imshow('Detected Circles', vis_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save visualization
cv2.imwrite('visualization.png', vis_image)