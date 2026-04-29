import os
import pandas as pd
import pydicom
import cv2
import numpy as np
from tqdm import tqdm

def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def process_vindr_data(csv_path, source_img_dir, dest_img_dir, dest_label_dir):
    df = pd.read_csv(csv_path)
    df_nodule = df[df['class_name'] == 'Nodule/Mass']
    
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    grouped = df_nodule.groupby('image_id')

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
        
        dest_img_path = os.path.join(dest_img_dir, f"{image_id}.png")
        cv2.imwrite(dest_img_path, img_array)

        label_path = os.path.join(dest_label_dir, f"{image_id}.txt")
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                x_center, y_center, w, h = convert_bbox_to_yolo(
                    row['x_min'], row['y_min'], row['x_max'], row['y_max'], 
                    img_width, img_height
                )
                f.write(f"0 {x_center} {y_center} {w} {h}\n")

if __name__ == '__main__':
    base_dir = "./data/raw"
    yolo_data_dir = "./data/yolo_format"
    train_csv = os.path.join(base_dir, "annotations/annotations_train.csv")
    test_csv = os.path.join(base_dir, "annotations/annotations_test.csv")
    train_img_src = os.path.join(base_dir, "train")
    test_img_src = os.path.join(base_dir, "test")

    process_vindr_data(train_csv, train_img_src, os.path.join(yolo_data_dir, "images/train"), os.path.join(yolo_data_dir, "labels/train"))
    process_vindr_data(test_csv, test_img_src, os.path.join(yolo_data_dir, "images/val"), os.path.join(yolo_data_dir, "labels/val"))