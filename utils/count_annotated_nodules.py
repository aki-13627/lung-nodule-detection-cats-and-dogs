import os
from pathlib import Path

def count_annotations(labels_dir):
    total_annotations = 0
    label_path = Path(labels_dir)
    
    if not label_path.exists():
        print(f"Directory not found: {labels_dir}")
        return 0
        
    for txt_file in label_path.glob("*.txt"):
        with open(txt_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            total_annotations += len(lines)
                
    return total_annotations

if __name__ == "__main__":
    train_dir = "yolo_dataset/labels/train"
    val_dir = "yolo_dataset/labels/val"
    
    train_nodules = count_annotations(train_dir)
    val_nodules = count_annotations(val_dir)
    
    print(f"Train結節総数: {train_nodules}")
    print(f"Val結節総数: {val_nodules}")