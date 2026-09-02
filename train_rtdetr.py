from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt")
    
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=512,
        batch=4,
        device=0,
        project="outputs_rtdetr",
        name="lung_nodule_run"
    )

if __name__ == "__main__":
    main()