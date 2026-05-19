import os
import json
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

val_img_dir = "./data/coco_format/images/train"
val_json_path = "./data/coco_format/annotations/instances_train.json"
checkpoint_path = "./deformable_detr_nodules/checkpoint-21100"
output_dir = "./inference_compare"
num_samples = 20

os.makedirs(output_dir, exist_ok=True)

with open(val_json_path, 'r') as f:
    coco_data = json.load(f)

img_name_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}
ann_by_img_id = {}
for ann in coco_data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in ann_by_img_id:
        ann_by_img_id[img_id] = []
    ann_by_img_id[img_id].append(ann)

positive_images = [img["file_name"] for img in coco_data["images"] if img["id"] in ann_by_img_id]
target_images = positive_images[:num_samples]

image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
model = AutoModelForObjectDetection.from_pretrained(checkpoint_path)
model.eval()

for img_name in target_images:
    image_path = os.path.join(val_img_dir, img_name)
    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")
    img_id = img_name_to_id[img_name]
    gt_anns = ann_by_img_id.get(img_id, [])

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    draw = ImageDraw.Draw(image)

    for ann in gt_anns:
        x, y, w, h = ann["bbox"]
        gt_box = [x, y, x + w, y + h]
        draw.rectangle(gt_box, outline="green", width=3)
        draw.text((x, max(0, y - 15)), "GT", fill="green")

    for score, box in zip(results["scores"], results["boxes"]):
        pred_box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(pred_box, outline="red", width=3)
        draw.text((pred_box[0], pred_box[1]), f"Pred: {round(score.item(), 3)}", fill="red")

    out_path = os.path.join(output_dir, f"compare_{img_name}")
    image.save(out_path)
    print(f"Saved: {out_path}")