from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt")
    
    results = model.train(
        data="data.yaml",
        lr0=1e-2,
        lrf=0.01,
        epochs=100,
        imgsz=768,
        batch=8,
        device=0,
        cls=2.5,
        box=7.5,
        mosaic=0.5,
        project="outputs_rtdetr",
        name="lung_nodule_recall_run"
    )

if __name__ == "__main__":
    main()