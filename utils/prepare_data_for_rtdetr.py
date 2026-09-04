import os
import shutil
import random
from pathlib import Path
from pycocotools.coco import COCO

def main():
    ann_file = "./merged_coco_annotations.json"
    output_dir = Path("yolo_dataset")
    
    # 既存フォルダがある場合はリセット（古いキャッシュやゴミを防ぐため）
    if output_dir.exists():
        shutil.rmtree(output_dir)

    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 陽性と陰性の画像をディレクトリから直接取得
    pos_dir = Path("data_png/with-nodules-or-masses")
    neg_dir = Path("data_png/no-nodules-or-masses")
    
    pos_images = [p.name for p in pos_dir.rglob('*.png')]
    neg_images = [p.name for p in neg_dir.rglob('*.png')]
    
    # 2. ランダムシャッフル（固定シード42で毎回同じ分割を再現）
    random.seed(42)
    random.shuffle(pos_images)
    random.shuffle(neg_images)
    
    # 3. 陽性画像の80%をtrain、20%をvalに分割
    train_pos_count = int(len(pos_images) * 0.8)
    train_pos = pos_images[:train_pos_count]
    val_pos = pos_images[train_pos_count:]
    
    # 4. 陰性画像の枚数を制限して抽出（Recall低下を防ぐため少なめに設定）
    # ※枚数はここで自由に変更可能です
    train_neg_count = 63  # trainに含める陰性（約25%程度）
    val_neg_count = 13    # valに含める陰性
    train_neg = neg_images[:train_neg_count]
    val_neg = neg_images[train_neg_count:train_neg_count + val_neg_count]
    
    train_files = set(train_pos + train_neg)
    val_files = set(val_pos + val_neg)
    
    # Textファイルも念のため再生成して上書き保存
    Path("outputs/split").mkdir(parents=True, exist_ok=True)
    with open("outputs/split/train.txt", "w") as f:
        f.write("\n".join(train_files))
    with open("outputs/split/val.txt", "w") as f:
        f.write("\n".join(val_files))

    # 5. 画像のコピーとYOLOラベルの作成
    image_paths = {path.name: path for path in Path("./data_png").rglob('*.png')}
    coco = COCO(ann_file)
    
    for img_id, img_info in coco.imgs.items():
        file_name = os.path.basename(img_info['file_name'])
        
        if file_name not in train_files and file_name not in val_files:
            continue
        if file_name not in image_paths:
            continue

        if file_name in train_files:
            img_dest = images_train / file_name
            label_dest = labels_train / f"{Path(file_name).stem}.txt"
        else:
            img_dest = images_val / file_name
            label_dest = labels_val / f"{Path(file_name).stem}.txt"

        shutil.copy(image_paths[file_name], img_dest)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        img_w = img_info['width']
        img_h = img_info['height']

        with open(label_dest, "w") as label_f:
            # 陰性画像の場合は空のtxtが生成され、YOLOが背景として認識します
            for ann in anns:
                x_min, y_min, w, h = ann['bbox']
                cx = (x_min + w / 2) / img_w
                cy = (y_min + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                label_f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                
    print(f"構築完了! Train: {len(train_files)}枚 (陽性:{len(train_pos)} 陰性:{len(train_neg)})")
    print(f"構築完了! Val  : {len(val_files)}枚 (陽性:{len(val_pos)} 陰性:{len(val_neg)})")

if __name__ == "__main__":
    main()