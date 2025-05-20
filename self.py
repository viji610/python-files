

import cv2
import numpy as np

# Load the shelf image
image = cv2.imread("shelf - Copy.jpg")
if image is None:
    print("Image not found.")
    exit()

# Resize image for consistency (optional)
image = cv2.resize(image, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection using Canny (detecting product edges)
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around detected contours (products)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:  # Ignore small contours (noise)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, 'Product', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Convert to HSV color space to separate color-based features
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Threshold for light colors (empty spaces tend to have lighter backgrounds)
lower_bound = np.array([0, 0, 200])  # Lower HSV values for light colors
upper_bound = np.array([255, 255, 255])  # Upper HSV values for light colors

# Mask to detect light regions (empty spaces)
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Find contours in the mask to detect empty spaces (background)
empty_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around detected empty spaces
for contour in empty_contours:
    area = cv2.contourArea(contour)
    if area > 1000:  # Ignore small empty contours
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, 'Empty Space', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Show the final result with detected products and empty spaces
cv2.imshow("Shelf Inventory Monitoring", image)
cv2.waitKey(0)
cv2.destroyAllWindows()