import os
import json
import torch
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

        target = {
            "image_id": img_id,
            "annotations": anns
        }

        encoding = self.image_processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": labels}

def main():
    checkpoint = "SenseTime/deformable-detr"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    
    id2label = {1: "Nodule/Mass"}
    label2id = {"Nodule/Mass": 1}
    
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
        
        batch_dict = image_processor.pad({"pixel_values": pixel_values}, return_tensors="pt")
        batch_dict["labels"] = labels
        return batch_dict

    training_args = TrainingArguments(
        output_dir="deformable_detr_nodules",
        per_device_train_batch_size=4,
        num_train_epochs=100,
        fp16=True,
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

if __name__ == '__main__':
    main()