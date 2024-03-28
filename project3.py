import cv2
import numpy as np
import os
import math
from skimage import measure
def histogram(x, image):
    # Check if width or height is too small
    if x[2] <= 0 or x[3] <= 0:
        return None

    roi = image[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]
    if roi.size == 0:  # Check if ROI is empty
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    return hist

def histogramUpper(x, image):
    # Check if width or height is too small
    if x[2] <= 0 or x[3] <= 0:
        return None

    # Adjust the range for height and width
    roi = image[x[1]:x[1]+((x[3])//2), x[0]:x[0]+x[2]]
    if roi.size == 0:  # Check if ROI is empty
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    return hist


def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images[os.path.splitext(filename)[0]] = img
    return images




from torchvision.ops import masks_to_boxes
from PIL import Image
def generate_histogram_coordinates_from_masks(folder):
    histogram_coordinates = []
    for filename in os.listdir(folder):
        if filename.endswith("_mask.png"):
            mask_path = os.path.join(folder, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
             # Thresholding to highlight the person (foreground)
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

           

            # Find contours in the mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            # Iterate over contours to get bounding boxes
            for contour in contours:
                # Calculate bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)
                histogram_coordinates.append((filename.split("_mask")[0], [x, y, w, h]))
    return histogram_coordinates

folder_path = './examples/output_cam0'
image1FirstFull = [1, 1, 40, 178]
image1FirstUpper = [1, 1, 40, 178//2]



image1 = cv2.imread('person_1.png')



histogram_coordinates = generate_histogram_coordinates_from_masks('examples/output_cam0')

histogram_names_first_image = [
    [(image1FirstFull,image1), (image1FirstUpper,image1)]
]

def compare(histogram_names):
    person_max_value = []
    for j in range (len(histogram_names)):

        # Compare histograms and find the maximum intersection for each person
        for i, (filename, coordinates) in enumerate(histogram_coordinates):
            

            image_cam= cv2.imread('./images/images/cam0/'+filename+'.png')

            try:
                intersections = [
                    (cv2.compareHist(np.float32(histogram(histogram_names[j][0][0], histogram_names[j][0][1])), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT)),
                    (cv2.compareHist(np.float32(histogram(histogram_names[j][1][0], histogram_names[j][1][1])), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT)),
                    (cv2.compareHist(np.float32(histogram(histogram_names[j][0][0], histogram_names[j][0][1])), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT)),
                    (cv2.compareHist(np.float32(histogram(histogram_names[j][1][0], histogram_names[j][1][1])), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT))
                ]
            except:
                continue
            #find max
            max_intersection = max(intersections)
           
            # Store person index, filename, and coordinates of the max intersection value
            person_max_value.append((i, filename, max_intersection, coordinates))

        # Sort the person_max_value list based on max intersection values in descending order
    sorted_person_max_value = sorted(person_max_value, key=lambda x: (x[2], x[3]), reverse=True)

        # Get the top 100 
    top_100_people = sorted_person_max_value[:100]
    print("Top 100 most similar people:")
    for person in top_100_people:
        print("Filename:", person[1], "Coordinates:", person[3])
    return top_100_people
    
def show_images_one_by_one(top_people, folder_path):
    window_name = "Image Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for person in top_people:
        filename, coordinates = person[1],  person[3]
        image_path = os.path.join(folder_path, filename + '.png')
        image = cv2.imread(image_path)

        if image is not None:
            x, y, w, h = coordinates
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow(window_name, image)
            cv2.waitKey(0)
        else:
            print(f"Failed to load image: {filename}")

    cv2.destroyAllWindows()

# Run comparison

# img=histogram(image1FirstFull,image1)
# img=histogram(image1FirstUpper,image1)
top_100_people = compare(histogram_names_first_image)
show_images_one_by_one(top_100_people, './images/images/cam0/')
