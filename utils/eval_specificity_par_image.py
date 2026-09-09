import os
from pathlib import Path
from ultralytics import RTDETR

def calculate_image_specificity_dynamic():
    model = RTDETR("./runs/detect/outputs_rtdetr/lung_nodule_recall_run-23/weights/best.pt")
    
    images_dir = Path("yolo_dataset/images/train")
    labels_dir = Path("yolo_dataset/labels/train")
    
    negative_image_paths = []
    
    for img_path in images_dir.glob("*.*"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
            
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        
        is_negative = False
        
        if not label_path.exists():
            is_negative = True
        elif label_path.stat().st_size == 0:
            is_negative = True
        else:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if len(content) == 0:
                    is_negative = True
                    
        if is_negative:
            negative_image_paths.append(str(img_path))
            
    if not negative_image_paths:
        print("陰性画像が見つかりませんでした。")
        return
        
    print(f"陰性画像を {len(negative_image_paths)} 枚検出しました。推論を開始します...")
    
    results = model.predict(
        source=negative_image_paths,
        imgsz=512,
        conf=0.175,
        iou=0.4,
        device="mps",
        stream=True
    )
    
    total_negative_images = len(negative_image_paths)
    true_negative_count = 0
    
    for r in results:
        if len(r.boxes) == 0:
            true_negative_count += 1
            
    specificity = (true_negative_count / total_negative_images) * 100
    
    print("--- 画像単位の評価結果 ---")
    print(f"陰性画像の総数: {total_negative_images}枚")
    print(f"正しく「異常なし」と判定した枚数 (True Negative): {true_negative_count}枚")
    print(f"過剰検知した枚数 (False Positive): {total_negative_images - true_negative_count}枚")
    print(f"画像単位の特異度: {specificity:.1f}%")

if __name__ == "__main__":
    calculate_image_specificity_dynamic()