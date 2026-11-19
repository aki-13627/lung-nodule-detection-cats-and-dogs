from ultralytics import RTDETR

def main():
    model_path = "./runs/detect/outputs_rtdetr/lung_nodule_recall_run-25/weights/best.pt"
    model = RTDETR(model_path)
    
    metrics = model.val(
        data="data.yaml",
        split="val",
        device="mps",
        project="outputs_rtdetr",
        name="val_lung_nodule_run"
    )
if __name__ == "__main__":
    main()