import os
import json
import pandas as pd
import pydicom
import cv2
import numpy as np
from tqdm import tqdm

def process_vindr_to_coco(csv_path, source_img_dir, dest_img_dir, json_out_path):
    df = pd.read_csv(csv_path)
    df_nodule = df[df['class_name'] == 'Nodule/Mass']
    
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_out_path), exist_ok=True)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "Nodule/Mass"}]
    }

    grouped = df_nodule.groupby('image_id')
    
    image_id_map = {}
    img_counter = 1
    ann_counter = 1

    for image_id, group in tqdm(grouped):
        dicom_path = os.path.join(source_img_dir, f"{image_id}.dicom")
        
        if not os.path.exists(dicom_path):
            continue

        dicom_data = pydicom.dcmread(dicom_path)
        img_array = dicom_data.pixel_array
        img_height, img_width = img_array.shape

        img_array = img_array - np.min(img_array)
        if np.max(img_array) != 0:
            img_array = img_array / np.max(img_array)
        img_array = (img_array * 255).astype(np.uint8)
        
        file_name = f"{image_id}.png"
        dest_img_path = os.path.join(dest_img_dir, file_name)
        cv2.imwrite(dest_img_path, img_array)

        if image_id not in image_id_map:
            image_id_map[image_id] = img_counter
            coco_data["images"].append({
                "id": img_counter,
                "file_name": file_name,
                "width": int(img_width),
                "height": int(img_height)
            })
            img_counter += 1

        current_img_id = image_id_map[image_id]

        for _, row in group.iterrows():
            x_min = float(row['x_min'])
            y_min = float(row['y_min'])
            x_max = float(row['x_max'])
            y_max = float(row['y_max'])
            
            width = x_max - x_min
            height = y_max - y_min
            area = width * height

            coco_data["annotations"].append({
                "id": ann_counter,
                "image_id": current_img_id,
                "category_id": 0,
                "bbox": [x_min, y_min, width, height],
                "area": area,
                "iscrowd": 0
            })
            ann_counter += 1

    with open(json_out_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

if __name__ == '__main__':
    base_dir = "./data/raw"
    coco_data_dir = "./data/coco_format"
    
    train_csv = os.path.join(base_dir, "annotations/annotations_train.csv")
    test_csv = os.path.join(base_dir, "annotations/annotations_test.csv")
    train_img_src = os.path.join(base_dir, "train")
    test_img_src = os.path.join(base_dir, "test")

    process_vindr_to_coco(
        train_csv, 
        train_img_src, 
        os.path.join(coco_data_dir, "images/train"), 
        os.path.join(coco_data_dir, "annotations/instances_train.json")
    )
    process_vindr_to_coco(
        test_csv, 
        test_img_src, 
        os.path.join(coco_data_dir, "images/val"), 
        os.path.join(coco_data_dir, "annotations/instances_val.json")
    )