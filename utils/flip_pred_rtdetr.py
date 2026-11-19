import cv2
import numpy as np
from pathlib import Path
from ultralytics import RTDETR

def replace_grid_block():
    # 既存のグリッド画像と、保存先のパス
    grid_img_path = "images/pred/val_predictions_rtdetr_grid_2.jpg"
    out_grid_path = "images/pred/val_predictions_rtdetr_grid_2_replaced.jpg" # 上書きを避けるため別名で保存
    
    # 置き換える対象の画像パス
    target_img_name = "00059799_Kitamura_Renge_01.png"
    target_img_path = Path("yolo_dataset/images/val") / target_img_name
    label_path = Path("yolo_dataset/labels/val") / (target_img_path.stem + ".txt")
    
    model_path = "./runs/detect/outputs_rtdetr/lung_nodule_recall_run-25/weights/best.pt"
    
    # 1. 既存のグリッド画像を読み込む
    grid_img = cv2.imread(grid_img_path)
    if grid_img is None:
        print(f"Error: グリッド画像 {grid_img_path} が見つかりません。")
        return

    # 2. 対象の元画像を読み込む
    img_orig = cv2.imread(str(target_img_path))
    if img_orig is None:
        print(f"Error: 置き換え対象画像 {target_img_path} が見つかりません。")
        return

    # 3. モデルの読み込みと設定
    model = RTDETR(model_path)
    conf_threshold = 0.3862
    target_size = (512, 512)
    
    h_orig, w_orig = img_orig.shape[:2]
    gt_img = img_orig.copy()
    pred_img = img_orig.copy()
    
    # 4. Ground Truthの描画
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
                    cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # 5. Predictionの描画
    results = model.predict(source=str(target_img_path), imgsz=512, conf=conf_threshold, device="mps", verbose=False)
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
            label = f"{conf:.2f}"
            cv2.putText(pred_img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # 6. リサイズとテキスト追加 (No.5, No.6)
    gt_resized = cv2.resize(gt_img, target_size)
    pred_resized = cv2.resize(pred_img, target_size)
    
    cv2.putText(gt_resized, "No.5", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.putText(pred_resized, "No.6", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    # 7. 左右に結合 (1024x512のペア画像作成)
    pair = cv2.hconcat([gt_resized, pred_resized])
    # GTとPredの中央区切り線
    cv2.line(pair, (target_size[0], 0), (target_size[0], target_size[1]), (255, 255, 255), 2)
    
    # 8. グリッド画像の該当部分 (2行目の左側) を上書き
    # 座標: y=512~1024, x=0~1024
    y_start, y_end = 512, 1024
    x_start, x_end = 0, 1024
    
    grid_img[y_start:y_end, x_start:x_end] = pair
    
    # 9. 上書きによって消えたり削れたりした周囲の区切り線（白線）を再描画
    # 中央の縦線 (No.5,6 と No.7,8 の境界)
    cv2.line(grid_img, (1024, 512), (1024, 1024), (255, 255, 255), 4)
    # 上の横線 (1行目と2行目の境界)
    cv2.line(grid_img, (0, 512), (1024, 512), (255, 255, 255), 4)
    # 下の横線 (2行目と3行目の境界)
    cv2.line(grid_img, (0, 1024), (1024, 1024), (255, 255, 255), 4)
    
    # 10. 保存
    cv2.imwrite(out_grid_path, grid_img)
    print(f"Saved {out_grid_path}")

if __name__ == "__main__":
    replace_grid_block()