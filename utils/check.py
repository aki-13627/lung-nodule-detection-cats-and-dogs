import os
import json
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

checkpoint_path = "./deformable_detr_nodules/checkpoint-21100"
test_img_dir = "./data/coco_format/images/test"
test_json_path = "./data/coco_format/annotations/instances_test.json"
output_dir = "./inference_results"
num_samples = 5

os.makedirs(output_dir, exist_ok=True)

with open(test_json_path, 'r') as f:
    coco_data = json.load(f)

annotated_image_ids = set([ann["image_id"] for ann in coco_data["annotations"]])
positive_images = [img["file_name"] for img in coco_data["images"] if img["id"] in annotated_image_ids]
target_images = positive_images[:num_samples]

image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
model = AutoModelForObjectDetection.from_pretrained(checkpoint_path)
model.eval()

for img_name in target_images:
    image_path = os.path.join(test_img_dir, img_name)
    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"Nodule: {round(score.item(), 3)}", fill="red")

    out_path = os.path.join(output_dir, f"pred_{img_name}")
    image.save(out_path)
    print(f"Saved: {out_path}")