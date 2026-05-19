import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer

class NoduleDataset(Dataset):
    def __init__(self, img_dir, json_path, image_processor):
        self.img_dir = img_dir
        self.image_processor = image_processor
        with open(json_path, 'r') as f:
            self.coco = json.load(f)

        self.images = {img["id"]: img for img in self.coco["images"]}
        self.annotations = {}
        for ann in self.coco["annotations"]:
            if ann["image_id"] not in self.annotations:
                self.annotations[ann["image_id"]] = []
            self.annotations[ann["image_id"]].append(ann)

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        anns = self.annotations.get(img_id, [])

        if len(anns) == 0:
            return self.__getitem__((idx + 1) % len(self.image_ids))

        valid_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            valid_anns.append(ann)

        if len(valid_anns) == 0:
            return self.__getitem__((idx + 1) % len(self.image_ids))

        target = {
            "image_id": img_id,
            "annotations": valid_anns
        }

        try:
            encoding = self.image_processor(images=image, annotations=target, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()
            labels = encoding["labels"][0]
            return {"pixel_values": pixel_values, "labels": labels}
        except Exception:
            return self.__getitem__((idx + 1) % len(self.image_ids))

def main():
    checkpoint = "SenseTime/deformable-detr"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    
    id2label = {0: "Nodule/Mass"}
    label2id = {"Nodule/Mass": 0}
    
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_dataset = NoduleDataset(
        img_dir="./data/coco_format/images/train",
        json_path="./data/coco_format/annotations/instances_train.json",
        image_processor=image_processor
    )

    val_dataset = NoduleDataset(
        img_dir="./data/coco_format/images/val",
        json_path="./data/coco_format/annotations/instances_val.json",
        image_processor=image_processor
    )

    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_h = max(img.shape[1] for img in pixel_values)
        max_w = max(img.shape[2] for img in pixel_values)

        padded_pixel_values = []
        pixel_mask = []

        for img in pixel_values:
            _, h, w = img.shape
            pad_h = max_h - h
            pad_w = max_w - w

            padded_img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
            padded_pixel_values.append(padded_img)

            mask = torch.zeros((max_h, max_w), dtype=torch.long)
            mask[:h, :w] = 1
            pixel_mask.append(mask)

        return {
            "pixel_values": torch.stack(padded_pixel_values),
            "pixel_mask": torch.stack(pixel_mask),
            "labels": labels
        }

    training_args = TrainingArguments(
    output_dir="deformable_detr_nodules",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    num_train_epochs=40,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    data_collator=collate_fn,
)

    trainer.train()

if __name__ == '__main__':
    main()