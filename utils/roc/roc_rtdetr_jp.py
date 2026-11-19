import os
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import RTDETR
from sklearn.metrics import roc_curve, auc

def extract_patient_id(file_path):
    return file_path.stem.split('_')[0]

def generate_patient_level_roc():
    model = RTDETR("./runs/detect/outputs_rtdetr/lung_nodule_recall_run-25/weights/best.pt")
    images_dir = Path("yolo_dataset/images/val")
    labels_dir = Path("yolo_dataset/labels/val")
    
    base_conf = 0.0001
    model_name = "RT-DETR"
    image_paths = [p for p in images_dir.glob("*.*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    patient_gt = {}
    patient_max_conf = {}
    
    for img_path in image_paths:
        pid = extract_patient_id(img_path)
        
        if pid not in patient_gt:
            patient_gt[pid] = 0
            patient_max_conf[pid] = 0.0
            
        label_path = labels_dir / (img_path.stem + ".txt")
        has_annotation = False
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        has_annotation = True
                        break
        if has_annotation:
            patient_gt[pid] = 1
            
        results = model.predict(source=str(img_path), imgsz=512, conf=base_conf, device="mps", verbose=False)
        
        img_max_conf = 0.0
        if len(results[0].boxes) > 0:
            confs = results[0].boxes.conf.cpu().numpy()
            img_max_conf = float(np.max(confs))
            
        patient_max_conf[pid] = max(patient_max_conf[pid], img_max_conf)
        
    pids = list(patient_gt.keys())
    y_true = np.array([patient_gt[pid] for pid in pids])
    y_scores = np.array([patient_max_conf[pid] for pid in pids])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_thresh = thresholds[best_idx]
    best_sens = tpr[best_idx]
    best_spec = 1 - fpr[best_idx]
    best_j = youden_index[best_idx]
    
    fig = plt.figure(figsize=(8, 8)) 
    plt.plot(fpr, tpr, color='b', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot(fpr[best_idx], tpr[best_idx], marker='o', color='r', markersize=8, label=f'Best Youden (J = {best_j:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance')
    
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(False)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/patient_level_roc_curve_rtdetr_jp.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved patient-level ROC curve to images/patient_level_roc_curve_rtdetr_jp.png (AUC: {roc_auc:.4f})")

if __name__ == "__main__":
    generate_patient_level_roc()