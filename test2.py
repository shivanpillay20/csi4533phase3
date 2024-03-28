import os
from PIL import Image
import torch
import numpy as np
from utils import model, tools
from torchvision.ops import masks_to_boxes

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    mask_np = np.array(mask) 
    boxes = masks_to_boxes(mask_np)
    return boxes.numpy().astype(np.int32)

def generate_histogram_coordinates_from_masks(folder):
    histogram_coordinates = []
    for filename in os.listdir(folder):
        if filename.endswith("_mask.png"):
            mask_path = os.path.join(folder, filename)
            mask = Image.open(mask_path)
            
            boxes = extract_bboxes(mask)
            for bbox in boxes:
                y1, x1, y2, x2 = bbox
                width = x2 - x1
                height = y2 - y1
                histogram_coordinates.append((filename.split("_mask")[0], [x1, y1, width, height]))
                print(x1,y1,width,height)
            
    return histogram_coordinates

# Main script
if __name__ == "__main__":
    source_path_dir_cam1 = "images/images/cam1"
    output_path_dir_cam1 = "examples/output_cam1"

    generate_histogram_coordinates_from_masks(output_path_dir_cam1)
