import numpy as np
import cv2
# Example file path
#mask_file = './examples/cam0/1637433772063999700.png.npy'
masks = np.load('./examples/cam0/1637433780110605400.png.npy')
masks = masks.astype(np.uint8)
# Convert mask to uint8 type (if it's not already)




# Example file paths
min_width = 35
min_height = 85
image =cv2.imread( './images/images/cam0/1637433780110605400.png')
for i,mask in enumerate(masks):
       mask_uint8 = mask.astype(np.uint8)
       points = cv2.findNonZero(mask_uint8)
       most_similar_bbox = cv2.boundingRect(points)
       if most_similar_bbox is not None:
        x, y, w, h = most_similar_bbox
        if w >= min_width and h >= min_height:  # Check if both width and height are above the minimum
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with the bounding box
cv2.imshow('Image with Bounding Box', image)
cv2.waitKey(0)

# min_width = 35
# min_height = 85
# mask_uint8 = mask_np.astype(np.uint8) 
# img_np = np.array(image)
# for mask in enumerate(mask_uint8):
        
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Draw bounding box for each contour on the original image
# for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#             if w >= min_width and h >= min_height:  # Check if both width and height are above the minimum
#                 cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)