from ultralytics import RTDETR

def main():
    model_path = "./runs/detect/outputs_rtdetr/lung_nodule_recall_run-15/weights/best.pt"
    model = RTDETR(model_path)
    
    metrics = model.val(
        data="data.yaml",
        split="train",
        conf=0.481,
        iou=0.5,    
        project="outputs_rtdetr",  
        name="val_for_train_lung_nodule_run-15"    
    )
if __name__ == "__main__":
    main()