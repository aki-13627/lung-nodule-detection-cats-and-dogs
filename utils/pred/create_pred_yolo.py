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
    
    # 高解像度設定といただいた閾値
    target_size = (1024, 1024)
    conf_threshold = 0.0389
    
    pairs = []
    
    # 1. まず全画像の推論とペア画像の作成を行う
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
                        
        # デバイスを mps に指定
        results = model.predict(source=str(img_path), imgsz=512, conf=conf_threshold, device="cpu ", verbose=False)
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                # 予測の枠線を太く (8)、confのテキストは描画しない
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 8)
                
        gt_resized = cv2.resize(gt_img, target_size)
        pred_resized = cv2.resize(pred_img, target_size)
        
        # --- テキスト(Noなど)の描画を削除 ---
        # local_idx = len(pairs) % 8
        # gt_num = local_idx * 2 + 1
        # pred_num = local_idx * 2 + 2
        # cv2.putText(gt_resized, f"No.{gt_num}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
        # cv2.putText(pred_resized, f"No.{pred_num}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5)
        # ------------------------------------
        
        pair = cv2.hconcat([gt_resized, pred_resized])
        cv2.line(pair, (target_size[0], 0), (target_size[0], target_size[1]), (255, 255, 255), 4)
        
        pairs.append(pair)

    # 2. Grid 2 の特定のブロック（インデックス18）をフリップ（差し替え）処理
    replace_target_idx = 18
    replace_img_name = "00059799_Kitamura_Renge_01.png"
    replace_img_path = images_dir / replace_img_name
    
    if len(pairs) > replace_target_idx and replace_img_path.exists():
        img_orig = cv2.imread(str(replace_img_path))
        if img_orig is not None:
            h_orig, w_orig = img_orig.shape[:2]
            gt_img = img_orig.copy()
            pred_img = img_orig.copy()
            
            label_path = labels_dir / (replace_img_path.stem + ".txt")
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
                            cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 8)
                            
            # デバイス mps で推論
            results = model.predict(source=str(replace_img_path), imgsz=512, conf=conf_threshold, device="cpu", verbose=False)
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 8)
            
            gt_resized = cv2.resize(gt_img, target_size)
            pred_resized = cv2.resize(pred_img, target_size)
            
            # --- 差し替え先のテキスト(No.5, No.6)の描画も削除 ---
            # cv2.putText(gt_resized, "No.5", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
            # cv2.putText(pred_resized, "No.6", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5)
            # ----------------------------------------------------
            
            pair = cv2.hconcat([gt_resized, pred_resized])
            cv2.line(pair, (target_size[0], 0), (target_size[0], target_size[1]), (255, 255, 255), 4)
            
            # 配列の該当箇所を上書き
            pairs[replace_target_idx] = pair
            print(f"Replaced Grid 2 block with image: {replace_img_name}")

    # 3. グリッドの構築と保存 (pred_new フォルダ)
    grid_count = 0
    os.makedirs('images/pred_new', exist_ok=True)
    
    for i in range(0, len(pairs), 8):
        batch = pairs[i:i+8]
        
        while len(batch) < 8:
            batch.append(np.zeros_like(pairs[0]))
            
        rows = []
        for j in range(0, 8, 2):
            row = cv2.hconcat([batch[j], batch[j+1]])
            cv2.line(row, (target_size[0]*2, 0), (target_size[0]*2, target_size[1]), (255, 255, 255), 8)
            rows.append(row)
            
        grid = cv2.vconcat(rows)
        for j in range(1, 4):
            cv2.line(grid, (0, target_size[1]*j), (target_size[0]*4, target_size[1]*j), (255, 255, 255), 8)
            
        # 拡張子を .png に設定
        out_filename = f"images/pred_new/val_predictions_yolo_grid_{grid_count}_jp.png"
        cv2.imwrite(out_filename, grid)
        print(f"Saved {out_filename}")
        grid_count += 1

if __name__ == "__main__":
    create_prediction_grids()