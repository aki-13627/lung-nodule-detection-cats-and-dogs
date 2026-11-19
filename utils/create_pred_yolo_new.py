import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def create_prediction_grids():
    model = YOLO("./runs/detect/outputs_yolo/lung_nodule_recall_run-2/weights/best.pt")
    
    images_dir = Path("yolo_dataset/images/val")
    labels_dir = Path("yolo_dataset/labels/val")
    
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(images_dir.glob(ext))
    image_paths = sorted(image_paths)
    
    # 出力画像の解像度を2倍（1024x1024）
    target_size = (1024, 1024)
    conf_threshold = 0.0389
    
    pairs = []
    
    for img_path in image_paths:
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue
            
        h_orig, w_orig = img_orig.shape[:2]
        
        gt_img = img_orig.copy()
        pred_img = img_orig.copy()
        
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = int((cx - bw / 2) * w_orig)
                        y1 = int((cy - bh / 2) * h_orig)
                        x2 = int((cx + bw / 2) * w_orig)
                        y2 = int((cy + bh / 2) * h_orig)
                        # 正解の枠線を太く (8)
                        cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 8)
                        
        results = model.predict(source=str(img_path), imgsz=512, conf=conf_threshold, device="cpu", verbose=False)
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                # 予測の枠線を太く (8) ※スコアのテキスト描画（putText）を削除
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 8)
                
        gt_resized = cv2.resize(gt_img, target_size)
        pred_resized = cv2.resize(pred_img, target_size)
        
        local_idx = len(pairs) % 8
        gt_num = local_idx * 2 + 1
        pred_num = local_idx * 2 + 2
        
        # No.の文字を大きく・太くし、位置を調整
        cv2.putText(gt_resized, f"No.{gt_num}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
        cv2.putText(pred_resized, f"No.{pred_num}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5)
        
        pair = cv2.hconcat([gt_resized, pred_resized])
        # ペアの中央の区切り線を太く (4)
        cv2.line(pair, (target_size[0], 0), (target_size[0], target_size[1]), (255, 255, 255), 4)
        
        pairs.append(pair)
        
    grid_count = 0
    
    # フォルダ名を pred_new に変更
    os.makedirs('images/pred_new', exist_ok=True)
    
    for i in range(0, len(pairs), 8):
        batch = pairs[i:i+8]
        
        while len(batch) < 8:
            batch.append(np.zeros_like(pairs[0]))
            
        rows = []
        for j in range(0, 8, 2):
            row = cv2.hconcat([batch[j], batch[j+1]])
            # 縦のグリッド区切り線を太く (8)
            cv2.line(row, (target_size[0]*2, 0), (target_size[0]*2, target_size[1]), (255, 255, 255), 8)
            rows.append(row)
            
        grid = cv2.vconcat(rows)
        for j in range(1, 4):
            # 横のグリッド区切り線を太く (8)
            cv2.line(grid, (0, target_size[1]*j), (target_size[0]*4, target_size[1]*j), (255, 255, 255), 8)
            
        # 保存先のパスを pred_new に変更
        out_filename = f"images/pred_new/val_predictions_yolo_grid_{grid_count}.png"
        cv2.imwrite(out_filename, grid)
        print(f"Saved {out_filename}")
        grid_count += 1

if __name__ == "__main__":
    create_prediction_grids()