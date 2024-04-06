import cv2
import numpy as np
import os
#for cam0
#directory = './examples/cam0'
#for cam1
directory = './examples/cam1'

def histogram(x, image):
  
    roi = image[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]
    if roi.size == 0:  # Check if ROI is empty
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    return hist

def histogramUpper(x, image):
  

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


def process_and_box(image_path,output_file):
    
        # Extract filename without extension
        print(image_path)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        print(filename)

        # Read the image
        masks = np.load(image_path)
        masks = masks.astype(np.uint8)
        # Convert mask to uint8 type (if it's not already)

        # Example file paths
        min_width = 35
        min_height = 85
        for i,mask in enumerate(masks):
            mask_uint8 = mask.astype(np.uint8)
            points = cv2.findNonZero(mask_uint8)
            most_similar_bbox = cv2.boundingRect(points)
            if most_similar_bbox is not None:
                x, y, w, h = most_similar_bbox
                if w >= min_width and h >= min_height:  # Check if both width and height are above the minimum
                    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    output_file.write(f"{filename.split('.')[0]},{x},{y},{w},{h}\n")


        # Print the coordinates of the bounding boxes
        #for coord in histogram_coordinates:
         #   print(f"Bounding box coordinates for {coord[0]} (x, y, width, height): {coord[1]}")

        # Draw the bounding boxes on the image
        for coord in histogram_coordinates:
            x, y, w, h = coord[1]
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

       

        return histogram_coordinates
   

    

    

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
image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.png', '.jpg', '.jpeg','.npy'))]

histogram_coordinates = []  # Initialize histogram_coordinates outside the loop

# Process each image, draw boxes, and accumulate histogram coordinates
# Open the output file for writing
#with open("coordinates.txt", "w") as output_file:
with open("coordinates1.txt", "w") as output_file:
    # Process each image and write coordinates to the text file
    for file in image_files:
        process_and_box(file, output_file)

# Now histogram_coordinates should contain the results from all images



def compare(histogram_names):
    person_max_value = []
    for j in range (len(histogram_names)):
        

        # Compare histograms and find the maximum intersection for each person
        for i, (filename, coordinates) in enumerate(histogram_coordinates):
            #for cam0
            #image_cam= cv2.imread('./images/images/cam0/'+filename[0:19]+'.png')
            #for cam1
            image_cam= cv2.imread('./images/images/cam1/'+filename[0:19]+'.png')

         

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
    
def save_images(top_people, folder_path, person_index, camera_name):
    # Create a directory to save the images for the person
    save_dir = os.path.join('./output_folder', camera_name, f'person_{person_index}_images')
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

def save_top_images_for_person(histogram_names, folder_path, person_index, camera_name):
    top_100_people = compare(histogram_names)
    save_images(top_100_people, folder_path, person_index, camera_name)


# Save results for cam0
save_top_images_for_person(histogram_names_first_image, './images/images/cam0', 1, 'cam0')
# save_top_images_for_person(histogram_names_second_image, './images/images/cam0', 2, 'cam0')
# save_top_images_for_person(histogram_names_third_image, './images/images/cam0', 3, 'cam0')
# save_top_images_for_person(histogram_names_fouth_image, './images/images/cam0', 4, 'cam0')
# save_top_images_for_person(histogram_names_fifth_image, './images/images/cam0', 5, 'cam0')

# # Save results for cam1
# save_top_images_for_person(histogram_names_first_image, './images/images/cam1', 1, 'cam1')
# save_top_images_for_person(histogram_names_second_image, './images/images/cam1', 2, 'cam1')
# save_top_images_for_person(histogram_names_third_image, './images/images/cam1', 3, 'cam1')
# save_top_images_for_person(histogram_names_fouth_image, './images/images/cam1', 4, 'cam1')
# save_top_images_for_person(histogram_names_fifth_image, './images/images/cam1', 5, 'cam1')


