import os
from pathlib import Path
from ultralytics import RTDETR

def calculate_image_sensitivity_dynamic():
    model = RTDETR("./runs/detect/outputs_rtdetr/lung_nodule_recall_run-23/weights/best.pt")
    
    images_dir = Path("yolo_dataset/images/val")
    labels_dir = Path("yolo_dataset/labels/val")
    
    positive_image_paths = []
    
    for img_path in images_dir.glob("*.*"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
            
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        
        is_positive = False
        
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if len(content) > 0:
                    is_positive = True
                    
        if is_positive:
            positive_image_paths.append(str(img_path))
            
    if not positive_image_paths:
        print("陽性画像が見つかりませんでした。")
        return
        
    print(f"陽性画像を {len(positive_image_paths)} 枚検出しました。推論を開始します...")
    
    results = model.predict(
        source=positive_image_paths,
        imgsz=512,
        conf=0.175,
        iou=0.4,
        device="mps",
        stream=True
    )
    
    total_positive_images = len(positive_image_paths)
    true_positive_count = 0
    
    for r in results:
        if len(r.boxes) > 0:
            true_positive_count += 1
            
    sensitivity = (true_positive_count / total_positive_images) * 100
    
    print("--- 画像単位の評価結果 ---")
    print(f"陽性画像の総数: {total_positive_images}枚")
    print(f"正しく「異常あり」と判定した枚数 (True Positive): {true_positive_count}枚")
    print(f"完全に見落とした枚数 (False Negative): {total_positive_images - true_positive_count}枚")
    print(f"画像単位の感度: {sensitivity:.1f}%")

if __name__ == "__main__":
    calculate_image_sensitivity_dynamic()