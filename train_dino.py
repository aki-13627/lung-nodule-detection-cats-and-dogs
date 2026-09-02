import os
import random
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from models.dino import (
    DINONoduleDetector, 
    HungarianMatcher, 
    SetCriterion, 
    build_backbone, 
    build_transformer
)

class LungNoduleCocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.image_paths = {path.name: path for path in self.img_dir.rglob('*.png')}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        file_name = os.path.basename(img_info['file_name'])
        
        img_path = self.image_paths[file_name]
        image = Image.open(img_path).convert("RGB")
        
        img_w, img_h = image.size

        boxes = []
        labels = []
        for ann in anns:
            x_min, y_min, width, height = ann['bbox']
            cx = (x_min + width / 2) / img_w
            cy = (y_min + height / 2) / img_h
            w = width / img_w
            h = height / img_h
            boxes.append([cx, cy, w, h])
            labels.append(ann['category_id'])

        if len(boxes) > 0:
            target_boxes = torch.tensor(boxes, dtype=torch.float32)
            target_labels = torch.tensor(labels, dtype=torch.int64)
        else:
            target_boxes = torch.empty((0, 4), dtype=torch.float32)
            target_labels = torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": target_boxes,
            "labels": target_labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, target

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch, num_epochs):
    model.train()
    criterion.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
    for images, targets in pbar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(images)
        
        losses = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)

def evaluate(model, criterion, dataloader, device):
    model.eval()
    criterion.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            losses = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict)

            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    img_dir = "../data_png"
    ann_file = "./coco_annotations.json"
    
    outputs_dir = Path("outputs")
    checkpoints_dir = outputs_dir / "checkpoints"
    split_dir = outputs_dir / "split"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = LungNoduleCocoDataset(img_dir=img_dir, ann_file=ann_file, transform=transform)

    with_nodules_indices = []
    no_nodules_indices = []

    for idx, img_id in enumerate(full_dataset.ids):
        img_info = full_dataset.coco.loadImgs(img_id)[0]
        file_name = os.path.basename(img_info['file_name'])
        full_path = str(full_dataset.image_paths[file_name])
        
        if 'with-nodules-or-masses' in full_path:
            with_nodules_indices.append(idx)
        else:
            no_nodules_indices.append(idx)

    random.seed(42)
    random.shuffle(with_nodules_indices)
    random.shuffle(no_nodules_indices)

    train_with_size = min(130, len(with_nodules_indices))
    train_no_size = min(70, len(no_nodules_indices))
    
    train_with_indices = with_nodules_indices[:train_with_size] * 2 
    train_no_indices = no_nodules_indices[:train_no_size]

    train_indices = train_with_indices + train_no_indices
    
    val_with_indices = with_nodules_indices[train_with_size:]
    val_no_indices = no_nodules_indices[train_no_size : train_no_size + 15]

    val_indices = val_with_indices + val_no_indices

    train_files = []
    for idx in train_indices:
        img_id = full_dataset.ids[idx]
        img_info = full_dataset.coco.loadImgs(img_id)[0]
        train_files.append(os.path.basename(img_info['file_name']))
        
    val_files = []
    for idx in val_indices:
        img_id = full_dataset.ids[idx]
        img_info = full_dataset.coco.loadImgs(img_id)[0]
        val_files.append(os.path.basename(img_info['file_name']))

    with open(split_dir / "train.txt", "w") as f:
        f.write("\n".join(train_files))
        
    with open(split_dir / "val.txt", "w") as f:
        f.write("\n".join(val_files))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    backbone = build_backbone()
    transformer = build_transformer()

    model = DINONoduleDetector(backbone=backbone, transformer=transformer, num_classes=2, num_queries=300, d_model=256)
    model.to(device)

    matcher = HungarianMatcher(cost_class=20, cost_bbox=5.0, cost_giou=2.0)
    
    weight_dict = {'loss_ce': 10, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    for i in range(5):
        weight_dict.update({f'loss_ce_{i}': 10, f'loss_bbox_{i}': 5.0, f'loss_giou_{i}': 2.0})

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, focal_alpha=0.95)
    criterion.to(device)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=1e-5, weight_decay=1e-4)

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, criterion, optimizer, train_dataloader, device, epoch, num_epochs)
        val_loss = evaluate(model, criterion, val_dataloader, device)
        print(f"Epoch {epoch}/{num_epochs} Completed | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n")
        
        checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pth"
        torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    main()