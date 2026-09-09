from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")
    
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=512,
        batch=8,
        device=0,
        cls=2.5,
        box=5.0,
        mosaic=1.0,
        project="outputs_yolo26",
        name="lung_nodule_recall_run"
    )

if __name__ == "__main__":
    main()