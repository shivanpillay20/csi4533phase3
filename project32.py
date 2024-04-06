import cv2
import numpy as np
import os

def histogram(x, image):
    roi = image[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    return hist
    #return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
#function to calculate upper part of cam images
def histogramUpper(x, image):
    roi = image[x[1]:x[1]+x[3]//2, x[0]:x[0]+x[2]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    #return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist
    #return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images[os.path.splitext(filename)[0]] = img
    return images

folder_path = './images/images/cam0'
#folder_path = './images/images/cam1'

# Load images from the folder
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
#extract file name and coordinates from labels.txt
with open('coordinates.txt', 'r') as file:
#with open('coordinates1.txt', 'r') as file:
    for line in file:
        values = line.strip().split(',')
        filename = values[0]
        coordinates = list(map(int, values[-4:]))
        histogram_coordinates.append((filename, coordinates))

def compare(histogram_names):
    person_max_value = []
    for j in range (len(histogram_names)):

        # Compare histograms and find the maximum intersection for each person
        for i, (filename, coordinates) in enumerate(histogram_coordinates):
            if filename not in image_sequence:
                # skip if 
                continue
            #cam0
            image_cam= cv2.imread('./images/images/cam0/'+filename+'.png')
            #cam1
            #image_cam= cv2.imread('./images/images/cam1/'+filename+'.png')

            intersections = [
                #compare test images with cam images and add to intersections array
                (cv2.compareHist(np.float32(histogram(histogram_names[j][0][0], histogram_names[j][0][1])), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT)),
                (cv2.compareHist(np.float32(histogram(histogram_names[j][1][0], histogram_names[j][1][1])), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT)),
                (cv2.compareHist(np.float32(histogram(histogram_names[j][0][0], histogram_names[j][0][1])), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT)),
                (cv2.compareHist(np.float32(histogram(histogram_names[j][1][0], histogram_names[j][1][1])), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT))

            ]
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
save_top_images_for_person(histogram_names_second_image, './images/images/cam0', 2, 'cam0')
save_top_images_for_person(histogram_names_third_image, './images/images/cam0', 3, 'cam0')
save_top_images_for_person(histogram_names_fouth_image, './images/images/cam0', 4, 'cam0')
save_top_images_for_person(histogram_names_fifth_image, './images/images/cam0', 5, 'cam0')

# # Save results for cam1
# save_top_images_for_person(histogram_names_first_image, './images/images/cam1', 1, 'cam1')
# save_top_images_for_person(histogram_names_second_image, './images/images/cam1', 2, 'cam1')
# save_top_images_for_person(histogram_names_third_image, './images/images/cam1', 3, 'cam1')
# save_top_images_for_person(histogram_names_fouth_image, './images/images/cam1', 4, 'cam1')
# save_top_images_for_person(histogram_names_fifth_image, './images/images/cam1', 5, 'cam1')
# def show_images_one_by_one(top_people, folder_path):
#     window_name = "Image Viewer"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
#     for person in top_people:
#         filename, coordinates = person[1],  person[3]
#         image_path = os.path.join(folder_path, filename + '.png')
#         image = cv2.imread(image_path)

#         if image is not None:
#             x, y, w, h = coordinates
#             cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#             cv2.imshow(window_name, image)
#             cv2.waitKey(0)
#         else:
#             print(f"Failed to load image: {filename}")

#     cv2.destroyAllWindows()
    


# top_100_people=compare(histogram_names_third_image)
# show_images_one_by_one(top_100_people, folder_path)
# print("First Person - Femme en veste blueue")
# print("Result - Manually Calculated: 84% Correspondance")

#second person 1st image
# print("Second Person 1st image - Femme en veste marron:")
# top_100_people=compare(histogram_names_second_first_image)
# show_images_one_by_one(top_100_people, folder_path)
# print("Second Person - Femme en veste marron")
# print("Result - Manually Calculated: 88% Correspondance")
    
#first person 2nd image

    
#second person 2nd image
# print("Second Person 2nd image - Femme en veste marron:")
# top_100_people=compare(histogram_names_second_second_image)
# show_images_one_by_one(top_100_people, folder_path)
# print("Second Person - Femme en veste marron:")
# print("Result - Manually Calculated: 94% Correspondance")

#third person 2nd image
# print("Third Person - Homme en veste blueue:")
# top_100_people=compare(histogram_names_third)
# show_images_one_by_one(top_100_people, folder_path)
# print("Third Person - Homme en veste blueue:")
# print("Result - Manually Calculated: 69% Correspondance")