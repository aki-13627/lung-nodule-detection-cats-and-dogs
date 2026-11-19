import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def extract_patient_id(file_path):
    return file_path.stem.split('_')[0]

def generate_patient_level_cm():
    model = YOLO("./runs/detect/outputs_yolo/lung_nodule_recall_run/weights/best.pt")
    images_dir = Path("yolo_dataset/images/val")
    labels_dir = Path("yolo_dataset/labels/val")
    
    threshold =  0.0389
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
            
        results = model.predict(source=str(img_path), imgsz=512, conf=threshold, device="cpu", verbose=False)
        
        img_max_conf = 0.0
        if len(results[0].boxes) > 0:
            confs = results[0].boxes.conf.cpu().numpy()
            img_max_conf = float(np.max(confs))
            
        patient_max_conf[pid] = max(patient_max_conf[pid], img_max_conf)
        
    pids = list(patient_gt.keys())
    y_true = np.array([patient_gt[pid] for pid in pids])
    y_pred = np.array([1 if patient_max_conf[pid] >= threshold else 0 for pid in pids])
    
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    cm_custom = cm[:, [1, 0]]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_custom)
    
    # 変更点: cmapをBluesに変更し、fontsizeを36に拡大
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d', colorbar=False, text_kw={'fontsize': 36})
    
    # 変更点: 軸のラベル（True label / Predicted label）とメモリを非表示にする
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    
    os.makedirs('images', exist_ok=True)
    save_path = 'images/patient_level_cm_rtdetr_jp.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved patient-level Confusion Matrix to {save_path}")

if __name__ == "__main__":
    generate_patient_level_cm()