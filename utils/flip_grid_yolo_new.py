import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def replace_grid_block_yolo_high_res():
    # 入出力を pred_new フォルダと .png に変更
    grid_img_path = "images/pred_new/val_predictions_yolo_grid_2.png"
    out_grid_path = "images/pred_new/val_predictions_yolo_grid_2_replaced.png"
    
    target_img_name = "00059799_Kitamura_Renge_01.png"
    target_img_path = Path("yolo_dataset/images/val") / target_img_name
    label_path = Path("yolo_dataset/labels/val") / (target_img_path.stem + ".txt")
    
    model_path = "./runs/detect/outputs_yolo/lung_nodule_recall_run-2/weights/best.pt"
    
    grid_img = cv2.imread(grid_img_path)
    if grid_img is None:
        print(f"Error: {grid_img_path} not found.")
        return

    img_orig = cv2.imread(str(target_img_path))
    if img_orig is None:
        print(f"Error: {target_img_path} not found.")
        return

    model = YOLO(model_path)
    conf_threshold = 0.0389
    
    # 高解像度用のサイズに変更
    target_size = (1024, 1024)
    
    h_orig, w_orig = img_orig.shape[:2]
    gt_img = img_orig.copy()
    pred_img = img_orig.copy()
    
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

    results = model.predict(source=str(target_img_path), imgsz=512, conf=conf_threshold, device="cpu", verbose=False)
    
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # 予測の枠線を太く (8)、confテキスト削除
            cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 8)

    gt_resized = cv2.resize(gt_img, target_size)
    pred_resized = cv2.resize(pred_img, target_size)
    
    # 高解像度用の太い文字と位置
    cv2.putText(gt_resized, "No.5", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
    cv2.putText(pred_resized, "No.6", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5)

    pair = cv2.hconcat([gt_resized, pred_resized])
    # ペアの中央の区切り線を太く (4)
    cv2.line(pair, (target_size[0], 0), (target_size[0], target_size[1]), (255, 255, 255), 4)
    
    # 高解像度版の座標 (2行目、1列目のペア)
    # y_start, y_end = 512, 1024 -> 1024, 2048 に倍増
    # x_start, x_end = 0, 1024 -> 0, 2048 に倍増
    y_start, y_end = 1024, 2048
    x_start, x_end = 0, 2048
    
    grid_img[y_start:y_end, x_start:x_end] = pair
    
    # 境界線を上書きして綺麗にする (太さ8に合わせる)
    cv2.line(grid_img, (2048, 1024), (2048, 2048), (255, 255, 255), 8) # 右の縦線
    cv2.line(grid_img, (0, 1024), (2048, 1024), (255, 255, 255), 8)    # 上の横線
    cv2.line(grid_img, (0, 2048), (2048, 2048), (255, 255, 255), 8)    # 下の横線
    
    cv2.imwrite(out_grid_path, grid_img)
    print(f"Saved {out_grid_path}")

if __name__ == "__main__":
    replace_grid_block_yolo_high_res()