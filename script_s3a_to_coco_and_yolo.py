"""author @shajibghosh"""
"""created on mar 15, 2024 02:18:35"""

import os
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
from ast import literal_eval

save_file_as = 'train_plus_validate_FPIC.csv' # e.g., train_plus_validate_FPIC.csv' or 'test_FPIC.csv'
parent_dir = os.getcwd()
data_dir = os.path.join(parent_dir, 'images_fpic') # e.g., images, images_fpic
annotation_dir = os.path.join(parent_dir, 'smd_anns_fpic')  # e.g., annotations, smd_anns_fpic
annotation_files = os.listdir(annotation_dir)
li = []
for filename in annotation_files:
    df = pd.read_csv(os.path.join(annotation_dir,filename), index_col=None, header=0)
    li.append(df)

d_frame = pd.concat(li, axis=0, ignore_index=True)  
ann_df = d_frame.drop(["Author","Timestamp", "Validated", "Defect"], axis=1)
ann_df.to_csv(save_file_as, index=False)
print("All single annotation files are merged into " + save_file_as)

def convert_to_coco_json(input_csv_path, output_json_path, data_dir):
    # Load the CSV file
    df = pd.read_csv(input_csv_path)

    # Initialize the COCO dataset structure
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    class_names = df["Designator"].unique()
    
    category_id_mapping = {name: idx + 1 for idx, name in enumerate(class_names)}
    
    for class_name, class_id in category_id_mapping.items():
        coco_dataset["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "none"
        })
    
    annotation_id = 1
    seen_images = set()
    for _, row in df.iterrows():
        image_file = row["Image File"]
        image_path = os.path.join(data_dir, image_file)
        
        if image_file not in seen_images:
            # Read image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Add image info
            coco_dataset["images"].append({
                "id": image_file,
                "width": width,
                "height": height,
                "file_name": image_file
            })
            seen_images.add(image_file)
        
        class_name = row["Designator"]
        vertices = literal_eval(row["Vertices"])
        segmentation = [coord for vertex in vertices for coord in vertex]
        
        coco_dataset["annotations"].append({
            "id": annotation_id,
            "image_id": image_file,
            "category_id": category_id_mapping[class_name],
            "segmentation": [segmentation],  # COCO expects a list of lists
            "iscrowd": 0
        })
        annotation_id += 1
    
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_dataset, json_file)


input_file_name = save_file_as
output_file_name = save_file_as.split('.csv')[0] + '_coco_format.json'

input_csv_path = os.path.join(parent_dir, input_file_name)
output_json_path = os.path.join(parent_dir, output_file_name)

convert_to_coco_json(input_csv_path, output_json_path, data_dir)

def coco_to_yolo_polygon(coco_json_path, output_dir):
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dimensions = {image['id']: (image['width'], image['height']) for image in coco_data['images']}

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        image_width, image_height = image_dimensions[image_id]
        category_id = annotation['category_id']

        # Assuming each annotation's segmentation contains multiple polygons
        for segmentation in annotation['segmentation']:
            normalized_coords = []

            if isinstance(segmentation, list):
                # Flatten the list if it's a list of lists
                if all(isinstance(item, list) for item in segmentation):
                    segmentation = [coord for sublist in segmentation for coord in sublist]
                # Now assuming segmentation is a flat list of coordinates
                if all(isinstance(coord, (int, float)) for coord in segmentation):
                    for i in range(0, len(segmentation), 2):
                        x = segmentation[i]
                        y = segmentation[i + 1]
                        nx = x / image_width
                        ny = y / image_height
                        normalized_coords.extend([nx, ny])

                    yolo_annotation = f"{category_id} " + " ".join(f"{coord:.6f}" for coord in normalized_coords) + "\n"
                    output_path = os.path.join(output_dir, f"{image_id.split('.png')[0]}.txt")

                    with open(output_path, 'a') as file:
                        file.write(yolo_annotation)
                else:
                    print(f"Segmentation coordinates are not all numbers for annotation ID {annotation['id']}.")
            else:
                print(f"Unexpected segmentation format for annotation ID {annotation['id']}.")
# Example usage
coco_json_path = output_json_path
output_dir = os.path.join(parent_dir, 'smd_fpic_anns_yolo')
# Create the output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
coco_to_yolo_polygon(coco_json_path, output_dir)