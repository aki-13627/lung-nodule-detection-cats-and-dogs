import os
import shutil
from pathlib import Path
from pycocotools.coco import COCO

def main():
    img_dir = Path("./data_png")
    ann_file = "./coco_annotations.json"
    train_txt = "outputs/split/train.txt"
    val_txt = "outputs/split/val.txt"

    output_dir = Path("yolo_dataset")
    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    with open(train_txt, "r") as f:
        train_files = set(line.strip() for line in f if line.strip())
    with open(val_txt, "r") as f:
        val_files = set(line.strip() for line in f if line.strip())

    coco = COCO(ann_file)
    image_paths = {path.name: path for path in img_dir.rglob('*.png')}

    for img_id, img_info in coco.imgs.items():
        file_name = os.path.basename(img_info['file_name'])
        if file_name not in image_paths:
            continue

        if file_name in train_files:
            img_dest = images_train / file_name
            label_dest = labels_train / f"{Path(file_name).stem}.txt"
        elif file_name in val_files:
            img_dest = images_val / file_name
            label_dest = labels_val / f"{Path(file_name).stem}.txt"
        else:
            continue

        shutil.copy(image_paths[file_name], img_dest)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_w = img_info['width']
        img_h = img_info['height']

        with open(label_dest, "w") as f:
            for ann in anns:
                x_min, y_min, w, h = ann['bbox']
                cx = (x_min + w / 2) / img_w
                cy = (y_min + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

if __name__ == "__main__":
    main()