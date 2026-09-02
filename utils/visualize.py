import os
import sys
from pathlib import Path
import random
import torch
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.ops import box_convert

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.dino import DINONoduleDetector, build_backbone, build_transformer

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img_dir = Path("../lung-rads-data/data_png")
    ann_file = "../lung-rads-data/coco_annotations.json"
    checkpoint_path = "outputs/checkpoints/epoch_050.pth"
    val_txt_path = "outputs/split/train.txt"
    
    # 複数枚保存するためのディレクトリを作成
    vis_dir = Path("outputs/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(ann_file)
    with open(val_txt_path, "r") as f:
        val_files = set(line.strip() for line in f if line.strip())

    # 結節あり・なしの画像を分けて取得
    pos_img_ids = []
    neg_img_ids = []
    
    for img_id, img_info in coco.imgs.items():
        file_name = os.path.basename(img_info['file_name'])
        if file_name in val_files:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                pos_img_ids.append(img_id)
            else:
                neg_img_ids.append(img_id)

    # 結節ありを5枚、結節なしを5枚（合計10枚）ランダムに選択
    random.seed(10)
    selected_img_ids = random.sample(pos_img_ids, min(5, len(pos_img_ids))) + \
                       random.sample(neg_img_ids, min(5, len(neg_img_ids)))

    image_paths = {path.name: path for path in img_dir.rglob('*.png')}

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    backbone = build_backbone()
    transformer = build_transformer()
    model = DINONoduleDetector(backbone=backbone, transformer=transformer, num_classes=2, num_queries=300, d_model=256)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"{len(selected_img_ids)}枚の画像を可視化し、{vis_dir} に保存します...")

    with torch.no_grad():
        for img_id in selected_img_ids:
            img_info = coco.loadImgs(img_id)[0]
            file_name = os.path.basename(img_info['file_name'])
            
            if file_name not in image_paths:
                continue
                
            img_path = image_paths[file_name]
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size

            img_tensor = transform(image).unsqueeze(0).to(device)

            outputs = model(img_tensor)
            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]

            probs = pred_logits.sigmoid()
            nodule_probs = probs[:, 1]
            
            CONF_THRESH = 0.5
            mask = nodule_probs > CONF_THRESH
            pred_boxes_filtered = pred_boxes[mask]
            probs_filtered = nodule_probs[mask]

            draw = ImageDraw.Draw(image)

            # 正解バウンディングボックスの描画（緑色）
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                x_min, y_min, width, height = ann['bbox']
                draw.rectangle([x_min, y_min, x_min + width, y_min + height], outline="green", width=3)

            # 予測バウンディングボックスの描画（赤色）
            pred_boxes_xyxy = box_convert(pred_boxes_filtered, 'cxcywh', 'xyxy')
            for i in range(len(pred_boxes_xyxy)):
                box = pred_boxes_xyxy[i]
                prob = probs_filtered[i].item()
                x1, y1, x2, y2 = box.tolist()
                
                x1_abs = x1 * img_w
                y1_abs = y1 * img_h
                x2_abs = x2 * img_w
                y2_abs = y2 * img_h
                
                draw.rectangle([x1_abs, y1_abs, x2_abs, y2_abs], outline="red", width=1)
                draw.text((x1_abs, y1_abs), f"{prob:.2f}", fill="red")

            output_path = vis_dir / f"res_{file_name}"
            image.save(output_path)
            print(f"保存完了: {output_path} (予測数: {len(pred_boxes_filtered)})")

if __name__ == "__main__":
    main()