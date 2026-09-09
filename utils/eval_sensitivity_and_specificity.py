import os
from pathlib import Path
import cv2
import torch
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

def calculate_nodule_metrics():
    model = RTDETR("./runs/detect/outputs_rtdetr/lung_nodule_recall_run-23/weights/best.pt")
    images_dir = Path("yolo_dataset/images/val")
    labels_dir = Path("yolo_dataset/labels/val")
    
    iou_threshold = 0.4
    conf_threshold = 0.7
    
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_images = 0
    
    for img_path in images_dir.glob("*.*"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
            
        total_images += 1
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
        
        results = model.predict(source=str(img_path), imgsz=512, conf=conf_threshold, device="mps", verbose=False)
        
        pred_boxes = []
        if len(results[0].boxes) > 0:
            pred_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            
        matched_gt = set()
        matched_pred = set()
        
        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
        
        tp = len(matched_gt)
        fn = len(gt_boxes) - tp
        fp = len(pred_boxes) - len(matched_pred)
        
        total_tp += tp
        total_fn += fn
        total_fp += fp
        
    sensitivity = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0
    precision = (total_tp / (total_tp + total_fp)) * 100 if (total_tp + total_fp) > 0 else 0
    fppi = total_fp / total_images if total_images > 0 else 0
    
    print(f"Total True Positives (TP): {total_tp}")
    print(f"Total False Negatives (FN): {total_fn}")
    print(f"Total False Positives (FP): {total_fp}")
    print(f"Nodule-level Sensitivity (Recall): {sensitivity:.1f}%")
    print(f"Precision: {precision:.1f}%")
    print(f"False Positives Per Image (FPPI): {fppi:.2f}")

if __name__ == "__main__":
    calculate_nodule_metrics()