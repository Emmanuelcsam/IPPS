import cv2
import numpy as np

# Sample statistics (replace with your actual values)
stats = {'Recall': 89.5, 'Precision': 92.3, 'F1-Score': 90.9}

# Create blank image for chart
chart_h, chart_w = 400, 600
chart = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 255

# Chart parameters
margin = 50
bar_width = 120
bar_spacing = 50
max_height = chart_h - 2 * margin
max_value = 105  # Scale to 105%

# Draw bars
x_pos = margin
for i, (label, value) in enumerate(stats.items()):
    # Calculate bar height
    bar_height = int((value / max_value) * max_height)
    
    # Determine color based on value
    if value > 80:
        color = (0, 255, 0)  # Green
    elif value > 60:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 0, 255)  # Red
    
    # Draw bar
    top_y = chart_h - margin - bar_height
    bottom_y = chart_h - margin
    cv2.rectangle(chart, (x_pos, top_y), (x_pos + bar_width, bottom_y), color, -1)
    
    # Add value label above bar
    text = f'{value:.1f}%'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x_pos + (bar_width - text_size[0]) // 2
    text_y = top_y - 10
    cv2.putText(chart, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add metric label below bar
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_x = x_pos + (bar_width - label_size[0]) // 2
    label_y = bottom_y + 25
    cv2.putText(chart, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    x_pos += bar_width + bar_spacing

# Add title
title = 'Detection Performance Metrics'
title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
title_x = (chart_w - title_size[0]) // 2
cv2.putText(chart, title, (title_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Draw axis
cv2.line(chart, (margin - 10, chart_h - margin), (chart_w - margin, chart_h - margin), (0, 0, 0), 2)

# Save and display
cv2.imwrite('stats_chart.jpg', chart)
cv2.imshow('Statistics Chart', chart)
cv2.waitKey(0)
cv2.destroyAllWindows()