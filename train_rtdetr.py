from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt")
    
    results = model.train(
        data="data.yaml",
        epochs=200,
        imgsz=1024,
        batch=2,
        device=0,
        project="outputs_rtdetr",
        name="lung_nodule_run"
    )

if __name__ == "__main__":
    main()