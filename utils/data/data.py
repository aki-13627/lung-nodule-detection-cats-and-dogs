from pathlib import Path
from collections import defaultdict
import random
import shutil
import statistics

SRC_DIR = Path("yolo_dataset")
DST_DIR = Path("yolo_dataset_01")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_DIR_NAME = "labels" if (SRC_DIR / "labels").exists() else "text"
VAL_RATIO = 0.2
SEED = 42
NUM_TRIALS = 5000


def get_case_id(filepath: Path) -> str:
    return filepath.stem.split("_")[0]


def get_label_path(image_path: Path, split: str) -> Path:
    return (SRC_DIR / LABEL_DIR_NAME / split / image_path.name).with_suffix(".txt")


def is_positive_label(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return label_path.read_text(encoding="utf-8").strip() != ""


def collect_cases() -> dict[str, list[tuple[Path, Path, bool]]]:
    cases = defaultdict(list)
    for split in ["train", "val"]:
        img_dir = SRC_DIR / "images" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                lbl_path = get_label_path(img_path, split)
                pos = is_positive_label(lbl_path)
                cases[get_case_id(img_path)].append((img_path, lbl_path, pos))
    return cases


def split_group_balanced(
    case_items: list[tuple[str, int]], val_ratio: float, rng: random.Random
) -> tuple[set[str], set[str]]:
    n_cases = len(case_items)
    n_val_cases = max(1, round(n_cases * val_ratio))
    total_imgs = sum(cnt for _, cnt in case_items)
    target_val_imgs = total_imgs * val_ratio

    best_val_ids = set()
    best_diff = float("inf")

    for _ in range(NUM_TRIALS):
        sampled = rng.sample(case_items, n_val_cases)
        val_imgs = sum(cnt for _, cnt in sampled)
        diff = abs(val_imgs - target_val_imgs)
        if diff < best_diff:
            best_diff = diff
            best_val_ids = {cid for cid, _ in sampled}
            if diff == 0:
                break

    all_ids = {cid for cid, _ in case_items}
    return all_ids - best_val_ids, best_val_ids


def copy_split(
    cases: dict[str, list[tuple[Path, Path, bool]]],
    train_ids: set[str],
    val_ids: set[str],
) -> None:
    if DST_DIR.exists():
        raise FileExistsError(f"{DST_DIR} already exists.")

    for split in ["train", "val"]:
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / LABEL_DIR_NAME / split).mkdir(parents=True, exist_ok=True)

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for cid in sorted(ids):
            for img_path, lbl_path, _ in cases[cid]:
                shutil.copy2(img_path, DST_DIR / "images" / split / img_path.name)
                if lbl_path.exists():
                    shutil.copy2(lbl_path, DST_DIR / LABEL_DIR_NAME / split / lbl_path.name)


def print_summary(
    cases: dict[str, list[tuple[Path, Path, bool]]],
    train_ids: set[str],
    val_ids: set[str],
) -> None:
    overlap = train_ids & val_ids
    print(f"Created: {DST_DIR} (overlap cases: {len(overlap)})")

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        pos_counts = [len(cases[cid]) for cid in ids if any(p for _, _, p in cases[cid])]
        neg_counts = [len(cases[cid]) for cid in ids if not any(p for _, _, p in cases[cid])]
        all_counts = [len(cases[cid]) for cid in ids]
        print(f"\n[{split}]")
        for name, counts in [("Positive", pos_counts), ("Negative", neg_counts), ("Total", all_counts)]:
            if not counts:
                print(f"  {name}: 0 cases")
                continue
            print(
                f"  {name}: {len(counts)} cases, {sum(counts)} images | "
                f"mean={statistics.mean(counts):.2f}, median={statistics.median(counts):.2f}"
            )


def main() -> None:
    rng = random.Random(SEED)
    cases = collect_cases()

    pos_cases = [(cid, len(items)) for cid, items in cases.items() if any(p for _, _, p in items)]
    neg_cases = [(cid, len(items)) for cid, items in cases.items() if not any(p for _, _, p in items)]

    pos_train, pos_val = split_group_balanced(pos_cases, VAL_RATIO, rng)
    neg_train, neg_val = split_group_balanced(neg_cases, VAL_RATIO, rng)

    train_ids = pos_train | neg_train
    val_ids = pos_val | neg_val

    copy_split(cases, train_ids, val_ids)
    print_summary(cases, train_ids, val_ids)


if __name__ == "__main__":
    main()