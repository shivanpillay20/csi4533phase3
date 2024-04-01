import cv2
import numpy as np
import os

directory = './examples/output_cam0'

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



def process_and_box(image_path, min_width=10, min_height=50, max_width=185, max_height=330, aspect_ratio=1.0):
    # Extract filename without extension
    try:
        filename = os.path.splitext(os.path.basename(image_path))[0]
    

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to read image '{image_path}'")
            return []

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
        return histogram_coordinates
    except Exception as e:
        print(f"Error processing image '{image_path}': {str(e)}")
        return []
        

    

    

folder_path = './images/images/cam0'
image_sequence = load_images_from_folder(folder_path)

image1FirstFull = [1, 1, 41, 178]
image1FirstUpper = [1, 1, 41, 178//2]

image2SecondFull=[1,1,75,250]
image2SecondUpper=[1,1,75,250//2]

image3ThirdFull=[1,1,58,202]
image3ThirdUpper=[1,1,58,202//2]

image4FourthFull=[1,1,74,275]
image4FourthUpper=[1,1,74,275//2]

image5FifthFull=[1, 1, 67, 252]
image5FifthUpper=[1, 1, 67, 252//2]

image1 = cv2.imread('person_1.png')
image2 = cv2.imread('person_2.png')
image3 = cv2.imread('person_3.png')
image4 = cv2.imread('person_4.png')
image5 = cv2.imread('person_5.png')


histogram_names_first_image = [
    [(image1FirstFull,image1), (image1FirstUpper,image1)]
]
histogram_names_second_image = [
    [(image2SecondFull,image2), (image2SecondUpper,image2)]
]
histogram_names_third_image = [
    [(image3ThirdFull,image3), (image3ThirdUpper,image3)]
]
histogram_names_fouth_image = [
    [(image4FourthFull,image4), (image4FourthUpper,image4)]
]
histogram_names_fifth_image = [
    [(image5FifthFull,image5), (image5FifthUpper,image5)]
]
histogram_coordinates = []

# List all image files in the directory
image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.png', '.jpg', '.jpeg'))]

histogram_coordinates = []  # Initialize histogram_coordinates outside the loop

# Process each image, draw boxes, and accumulate histogram coordinates
for file in image_files:
    histogram_coordinates.extend(process_and_box(file))  # Extend the list with results from each iteration

# Now histogram_coordinates should contain the results from all images



def compare(histogram_names):
    person_max_value = []
    for j in range (len(histogram_names)):
        

        # Compare histograms and find the maximum intersection for each person
        for i, (filename, coordinates) in enumerate(histogram_coordinates):
           
            image_cam= cv2.imread('./images/images/cam0/'+filename[0:19]+'.png')

         

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
    
def save_images(top_people, folder_path, person_index):
    # Create a directory to save the images for the person
    save_dir = os.path.join(folder_path, f'person_{person_index}_images')
    os.makedirs(save_dir, exist_ok=True)

    for i, person in enumerate(top_people):
        filename, coordinates = person[1],  person[3]
        image_path = os.path.join(folder_path, filename[0:19] + '.png')
        image = cv2.imread(image_path)

        if image is not None:
            x, y, w, h = coordinates
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Save the image with bounding box
            save_path = os.path.join(save_dir, f'{i+1}_{filename[0:19]}.png')
            cv2.imwrite(save_path, image)
            print(f"Saved image: {save_path}")
        else:
            print(f"Failed to load image: {filename}")

def save_top_images_for_person(histogram_names, folder_path, person_index):
    top_100_people = compare(histogram_names)
    save_images(top_100_people, folder_path, person_index)

# Example usage
#save_top_images_for_person(histogram_names_first_image, './examples/output_cam0', 1)
#save_top_images_for_person(histogram_names_second_image, './examples/output_cam0', 2)
#save_top_images_for_person(histogram_names_third_image, './examples/output_cam0', 3)
#save_top_images_for_person(histogram_names_fouth_image, './images/images/cam0', 4)
save_top_images_for_person(histogram_names_fifth_image, './images/images/cam0', 5)

# Run comparison

# top_100_people = compare(histogram_names_first_image)
# show_images_one_by_one(top_100_people, './examples/output_cam0')
# top_100_people = compare(histogram_names_second_image)
# show_images_one_by_one(top_100_people, './examples/output_cam0')
# top_100_people = compare(histogram_names_third_image)
# show_images_one_by_one(top_100_people, './examples/output_cam0')
# top_100_people = compare(histogram_names_fouth_image)
# show_images_one_by_one(top_100_people, './examples/output_cam0')

#top_100_people = compare(histogram_names_fifth_image)
#show_images_one_by_one(top_100_people, './examples/output_cam0')
