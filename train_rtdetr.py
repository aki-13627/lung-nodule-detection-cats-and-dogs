from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt")
    
    model.tune(
        data="data.yaml",
        epochs=30,
        iterations=30,
        imgsz=768,
        batch=4,
        device=0,
        project="outputs_rtdetr",
        name="tune_lung_nodule"
    )

if __name__ == "__main__":
    main()