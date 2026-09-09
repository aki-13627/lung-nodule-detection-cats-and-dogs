import os
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import RTDETR

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def generate_pr_curve():
    model = RTDETR("./runs/detect/outputs_rtdetr/lung_nodule_recall_run-23/weights/best.pt")
    images_dir = Path("yolo_dataset/images/val")
    labels_dir = Path("yolo_dataset/labels/val")
    
    iou_threshold = 0.4
    base_conf = 0.0001
    
    all_gt_boxes = []
    all_pred_boxes = []
    total_gt = 0
    
    image_paths = [p for p in images_dir.glob("*.*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        gt_boxes = []
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        gt_boxes.append([x1, y1, x2, y2])
        all_gt_boxes.append(gt_boxes)
        total_gt += len(gt_boxes)
        
        results = model.predict(source=str(img_path), imgsz=512, conf=base_conf, device="mps", verbose=False)
        
        pred_boxes = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            confs = results[0].boxes.conf.cpu().numpy().tolist()
            for b, c in zip(boxes, confs):
                pred_boxes.append(b + [c])
        all_pred_boxes.append(pred_boxes)
        
    flat_preds = []
    for img_idx, preds in enumerate(all_pred_boxes):
        for p in preds:
            flat_preds.append({'conf': p[4], 'img_idx': img_idx, 'box': p[:4]})
            
    flat_preds.sort(key=lambda x: x['conf'], reverse=True)
    
    tp = np.zeros(len(flat_preds))
    fp = np.zeros(len(flat_preds))
    matched_gt = {i: set() for i in range(len(all_gt_boxes))}
    
    
    for i, pred in enumerate(flat_preds):
        img_idx = pred['img_idx']
        p_box = pred['box']
        gt_boxes = all_gt_boxes[img_idx]
        
        best_iou = 0
        best_gt_idx = -1
        
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt[img_idx]:
                continue
            iou = calculate_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
                
        if best_iou >= iou_threshold:
            matched_gt[img_idx].add(best_gt_idx)
            tp[i] = 1
        else:
            fp[i] = 1
            
    # 4. 累積和を計算
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    # 5. 各スコア時点でのPrecisionとRecallを計算
    recalls = cum_tp / total_gt if total_gt > 0 else np.zeros_like(cum_tp)
    precisions = cum_tp / (cum_tp + cum_fp)
    
    # グラフの端点（Recall=0と1の時）を綺麗に描画するための処理
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    
    # 6. Interpolated（補間）Precisionの計算（単調減少にする）
    precisions_interp = np.maximum.accumulate(precisions[::-1])[::-1]
    
    # --- プロット処理 ---
    plt.figure(figsize=(8, 6))
    # markerを外し、単一の線（linestyle='-'）として描画
    plt.plot(recalls, precisions_interp, linestyle='-', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    
    plt.savefig('pr_curve_interpolated.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Precision-Recall curve to pr_curve_interpolated.png")

if __name__ == "__main__":
    generate_pr_curve()