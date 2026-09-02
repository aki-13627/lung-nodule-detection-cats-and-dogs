import os
import torch
import sys
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.dino import DINONoduleDetector, build_backbone, build_transformer

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    img_dir = Path("./data_png")
    ann_file = "./coco_annotations.json"
    checkpoint_path = "outputs/checkpoints/dino_model_final_20260902_133224.pth"
    val_txt_path = "outputs/split/train.txt"
    
    with open(val_txt_path, "r") as f:
        val_files = set(line.strip() for line in f if line.strip())
        
    print(f"Loading annotations from {ann_file}...")
    coco = COCO(ann_file)
    
    val_img_ids = []
    for img_id, img_info in coco.imgs.items():
        if os.path.basename(img_info['file_name']) in val_files:
            val_img_ids.append(img_id)
            
    image_paths = {path.name: path for path in img_dir.rglob('*.png')}

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    backbone = build_backbone()
    transformer = build_transformer()
    model = DINONoduleDetector(backbone=backbone, transformer=transformer, num_classes=2, num_queries=100, d_model=256)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    CONF_THRESH = 0.3
    IOU_THRESH = 0.5

    img_TP = 0
    img_FP = 0
    img_TN = 0
    img_FN = 0

    box_TP = 0
    box_FP = 0
    box_FN = 0

    with torch.no_grad():
        for img_id in tqdm(val_img_ids, desc="Evaluating"):
            img_info = coco.loadImgs(img_id)[0]
            file_name = os.path.basename(img_info['file_name'])
            
            if file_name not in image_paths:
                continue
                
            img_path = image_paths[file_name]
            image = Image.open(img_path).convert("RGB")
            
            img_w, img_h = image.size
            img_tensor = transform(image).unsqueeze(0).to(device)

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            gt_boxes = []
            for ann in anns:
                x_min, y_min, width, height = ann['bbox']
                cx = (x_min + width / 2) / img_w
                cy = (y_min + height / 2) / img_h
                w = width / img_w
                h = height / img_h
                gt_boxes.append([cx, cy, w, h])
                
            has_gt = len(gt_boxes) > 0
            if has_gt:
                gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32).to(device)
                gt_boxes_xyxy = box_convert(gt_boxes, 'cxcywh', 'xyxy')
            else:
                gt_boxes_xyxy = torch.empty((0, 4), dtype=torch.float32).to(device)

            outputs = model(img_tensor)
            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]

            probs = pred_logits.sigmoid()
            nodule_probs = probs[:, 1]
            
            mask = nodule_probs > CONF_THRESH
            pred_boxes_filtered = pred_boxes[mask]
            
            has_pred = len(pred_boxes_filtered) > 0

            if has_gt and has_pred:
                img_TP += 1
            elif has_gt and not has_pred:
                img_FN += 1
            elif not has_gt and has_pred:
                img_FP += 1
            else:
                img_TN += 1

            if has_pred:
                pred_boxes_xyxy = box_convert(pred_boxes_filtered, 'cxcywh', 'xyxy')
            else:
                pred_boxes_xyxy = torch.empty((0, 4), dtype=torch.float32).to(device)

            if has_gt and has_pred:
                ious = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                
                gt_matched = torch.zeros(len(gt_boxes_xyxy), dtype=torch.bool)
                pred_matched = torch.zeros(len(pred_boxes_xyxy), dtype=torch.bool)
                
                for p_idx in range(len(pred_boxes_xyxy)):
                    best_gt_idx = -1
                    best_iou = IOU_THRESH
                    for g_idx in range(len(gt_boxes_xyxy)):
                        if not gt_matched[g_idx] and ious[p_idx, g_idx] >= best_iou:
                            best_iou = ious[p_idx, g_idx]
                            best_gt_idx = g_idx
                    if best_gt_idx >= 0:
                        gt_matched[best_gt_idx] = True
                        pred_matched[p_idx] = True
                        box_TP += 1
                
                box_FP += (~pred_matched).sum().item()
                box_FN += (~gt_matched).sum().item()
            elif has_pred and not has_gt:
                box_FP += len(pred_boxes_xyxy)
            elif not has_pred and has_gt:
                box_FN += len(gt_boxes_xyxy)

    print("\n" + "="*40)
    print(" Evaluation Results (Score Thresh: 0.5, IoU Thresh: 0.5)")
    print("="*40 + "\n")
    
    print("[Image Level]")
    img_total = img_TP + img_TN + img_FP + img_FN
    print(f"TP: {img_TP}, TN: {img_TN}, FP: {img_FP}, FN: {img_FN} (Total: {img_total})")
    img_sensitivity = img_TP / (img_TP + img_FN) if (img_TP + img_FN) > 0 else 0.0
    img_specificity = img_TN / (img_TN + img_FP) if (img_TN + img_FP) > 0 else 0.0
    print(f"Image Level Sensitivity : {img_sensitivity:.4f}")
    print(f"Image Level Specificity : {img_specificity:.4f}\n")

    print("[Bounding Box Level]")
    print(f"TP: {box_TP}, FP: {box_FP}, FN: {box_FN}")
    box_sensitivity = box_TP / (box_TP + box_FN) if (box_TP + box_FN) > 0 else 0.0
    box_precision = box_TP / (box_TP + box_FP) if (box_TP + box_FP) > 0 else 0.0
    print(f"Box Level Recall      : {box_sensitivity:.4f}")
    print(f"Box Level Precision   : {box_precision:.4f}")

if __name__ == "__main__":
    main()