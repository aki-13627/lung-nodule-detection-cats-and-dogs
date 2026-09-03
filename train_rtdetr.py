from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt")
    
    
    search_space = {
        "lr0": (1e-5, 5e-4),,
        "lrf": (0.01, 1.0),
        "box": (1.0, 10.0),
        "cls": (0.5, 4.0),
        "scale": (0.0, 0.15),
        "degrees": (0.0, 10.0),
        "mosaic": (0.0, 0.5),


        "hsv_h": (0.0, 0.0),
        "hsv_s": (0.0, 0.0),
        "mixup": (0.0, 0.0),
        "cutmix": (0.0, 0.0),
        "flipud": (0.0, 0.0),
        "perspective": (0.0, 0.0),
        "shear": (0.0, 0.0)
    }
    
    model.tune(
        data="data.yaml",
        epochs=30,
        iterations=30,
        imgsz=768,
        batch=8,
        device=0,
        optimizer="AdamW",
        space=search_space,
        project="outputs_rtdetr",
        name="tune_lung_nodule_focused"
    )

if __name__ == "__main__":
    main()