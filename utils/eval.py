import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

val_img_dir = "./data/coco_format/images/train"
val_json_path = "./data/coco_format/annotations/instances_train.json"
checkpoint_path = "./deformable_detr_nodules/checkpoint-21100"
output_json_path = "./inference_results/predictions.json"

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
model = AutoModelForObjectDetection.from_pretrained(checkpoint_path).to(device)
model.eval()

coco_gt = COCO(val_json_path)
image_ids = coco_gt.getImgIds()

predictions = []


for img_id in tqdm(image_ids):
    img_info = coco_gt.loadImgs(img_id)[0]
    image_path = os.path.join(val_img_dir, img_info["file_name"])

    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.01)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.cpu().numpy().tolist()
        x, y, xmax, ymax = box
        w = xmax - x
        h = ymax - y

        predictions.append({
            "image_id": img_id,
            "category_id": int(label.cpu().item()),
            "bbox": [x, y, w, h],
            "score": float(score.cpu().item())
        })

with open(output_json_path, 'w') as f:
    json.dump(predictions, f)
print(f"Predictions saved to {output_json_path}")

if len(predictions) > 0:
    print("\nCalculating mAP...")
    coco_dt = coco_gt.loadRes(output_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
else:
    print("No objects were detected. Cannot calculate mAP.")