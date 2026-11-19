from pathlib import Path
from collections import defaultdict
import statistics

DATASET_DIR = Path("yolo_dataset")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_DIR_NAME = "labels" if (DATASET_DIR / "labels").exists() else "text"


def get_case_id(filepath: Path) -> str:
    return filepath.stem.split("_")[0]


def is_positive_image(image_path: Path, split: str) -> bool:
    label_path = (DATASET_DIR / LABEL_DIR_NAME / split / image_path.name).with_suffix(".txt")
    if not label_path.exists():
        return False
    return label_path.read_text(encoding="utf-8").strip() != ""


def summarize_split(split: str) -> None:
    img_dir = DATASET_DIR / "images" / split
    if not img_dir.exists():
        print(f"[{split}] directory not found: {img_dir}")
        return

    cases = defaultdict(list)
    sample_pairs = []

    for img_path in sorted(img_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
            case_id = get_case_id(img_path)
            pos = is_positive_image(img_path, split)
            cases[case_id].append(pos)
            if len(sample_pairs) < 5:
                sample_pairs.append((img_path.name, case_id))

    pos_counts = [len(flags) for flags in cases.values() if any(flags)]
    neg_counts = [len(flags) for flags in cases.values() if not any(flags)]
    all_counts = [len(flags) for flags in cases.values()]

    print(f"\n[{split}]")
    print("  sample (filename -> case_id):", sample_pairs)
    for label, counts in [("Positive", pos_counts), ("Negative", neg_counts), ("Total", all_counts)]:
        if not counts:
            print(f"  {label}: 0 cases")
            continue
        single = sum(c == 1 for c in counts)
        print(
            f"  {label}: {len(counts)} cases, {sum(counts)} images | "
            f"mean={statistics.mean(counts):.2f}, median={statistics.median(counts):.2f}, "
            f"min={min(counts)}, max={max(counts)}, single={single} ({single / len(counts) * 100:.1f}%)"
        )


for split in ["train", "val"]:
    summarize_split(split)