import cv2
import numpy as np
import os

# Directory where the images are stored
directory = './examples/output_cam0'

import cv2
import numpy as np
import os

def process_and_box(image_path, min_width=10, min_height=50, max_width=185, max_height=330, aspect_ratio=1.0):
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Read the image
    image = cv2.imread(image_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Morphological operations
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 15)
    mask = cv2.dilate(mask, kernel, iterations = 15)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to hold valid bounding boxes
    histogram_coordinates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Calculate the aspect ratio and area for filtering
        current_aspect_ratio = float(h) / w if w != 0 else 0
        current_area = w * h
        # Check if the contour meets the criteria including max width and height
        if min_width <= w <= max_width and min_height <= h <= max_height and current_aspect_ratio > aspect_ratio:
            histogram_coordinates.append((filename, [x, y, w, h]))

    # Print the coordinates of the bounding boxes
    for coord in histogram_coordinates:
        print(f"Bounding box coordinates for {coord[0]} (x, y, width, height): {coord[1]}")

    # Draw the bounding boxes on the image
    for coord in histogram_coordinates:
        x, y, w, h = coord[1]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image
    # cv2.imshow(f'Image {filename}', image)
    # cv2.waitKey(0)  # Wait for any key press to close the displayed image
    # cv2.destroyAllWindows()  # Close the image window

    return histogram_coordinates


# List all image files in the directory
image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Process each image, draw boxes, and display them one by one
for file in image_files:
    process_and_box(file)
