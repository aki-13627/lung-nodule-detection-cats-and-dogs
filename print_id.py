from pathlib import Path

def print_train_ids():
    images_dir = Path("yolo_dataset/images/train")
    unique_ids = set()
    
    if not images_dir.exists():
        print(f"Error: {images_dir} が見つかりません。")
        return
    
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for img_path in images_dir.glob(ext):
            patient_id = img_path.stem.split('_')[0]
            unique_ids.add(patient_id)
   
    sorted_ids = sorted(list(unique_ids), reverse=True)
    
    print("id, species")
    for patient_id in sorted_ids:
        print(f"{patient_id}, ")

if __name__ == "__main__":
    print_train_ids()