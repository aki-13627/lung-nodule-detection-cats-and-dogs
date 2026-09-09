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
    model = RTDETR("./runs/detect/outputs_rtdetr/lung_nodule_recall_run-23/weights/best.pt")
    images_dir = Path("yolo_dataset/images/val")
    labels_dir = Path("yolo_dataset/labels/val")
    
    base_conf = 0.0001
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
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance')
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Patient-Level ROC Curve')
    plt.grid(True)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc='lower right')
    
    plt.savefig('images/patient_level_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved patient-level ROC curve to images/patient_level_roc_curve.png (AUC: {roc_auc:.4f})")

if __name__ == "__main__":
    generate_patient_level_roc()