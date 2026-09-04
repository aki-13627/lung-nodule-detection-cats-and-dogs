from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt")
    
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=768,
        batch=8,
        device=0,
        cls=2.5,
        box=5.0,
        mosaic=1.0,
        project="outputs_rtdetr",
        name="lung_nodule_recall_run"
    )

if __name__ == "__main__":
    main()