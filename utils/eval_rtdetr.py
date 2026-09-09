from ultralytics import YOLO

def main():
    model_path = "./runs/detect/outputs_yolo/lung_nodule_recall_run/weights/best.pt"
    model = YOLO(model_path)
    
    metrics = model.val(
        data="data.yaml",
        split="val",
        device="cpu",
        imgsz=512,
        conf=0.2,
        iou=0.4,
        project="outputs_yolo",
        name="val_lung_nodule_run"
    )
if __name__ == "__main__":
    main()