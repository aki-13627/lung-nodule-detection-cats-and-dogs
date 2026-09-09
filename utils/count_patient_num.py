import os
from pathlib import Path

def count_unique_patients():
    images_dir = Path("yolo_dataset/images/val")
    patient_ids = set()

    for img_path in images_dir.glob("*.*"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
            
        filename = img_path.stem
        patient_id = filename.split('_')[0]
        patient_ids.add(patient_id)
        
    print(f"Total images: {len(list(images_dir.glob('*.*')))}")
    print(f"Total unique patients: {len(patient_ids)}")

if __name__ == "__main__":
    count_unique_patients()