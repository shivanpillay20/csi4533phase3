import os
from PIL import Image
from utils import model, tools
import torch


def process_images(source_path_dir, output_path_dir):
    # Create output directory 
    if not os.path.exists(output_path_dir):
        os.makedirs(output_path_dir)

    seg_model, transforms = model.get_model()

    # Iterate over images in cam0 /cam1
    for image_name in os.listdir(source_path_dir):
        if image_name.endswith(".png"):
            # Charger le modèle et appliquer les transformations à l'image
            image_path = os.path.join(source_path_dir, image_name)
            image = Image.open(image_path)
            transformed_img = transforms(image)

           # Effectuer l'inférence sur l'image transformée sans calculer les gradients
            with torch.no_grad():
                output = seg_model([transformed_img])

            
            result = tools.process_inference(output, image)

            # Save mask
            mask_name = image_name.split(".")[0] + "_mask.png"
            result.save(os.path.join(output_path_dir, mask_name))

# Main script
if __name__ == "__main__":
    
    source_path_dir_cam0 = "images/images/cam0"
    source_path_dir_cam1 = "images/images/cam1"
    
    output_path_dir_cam0 = "examples/output_cam0"
    output_path_dir_cam1 = "examples/output_cam1"
    

    # Process images for camera 0
    #process_images(source_path_dir_cam0, output_path_dir_cam0)

    # Process images for camera 01
    #process_images(source_path_dir_cam1, output_path_dir_cam1)

    
    # Process images for camera 01
    process_images(source_path_dir_cam01, output_path_dir_cam01)
