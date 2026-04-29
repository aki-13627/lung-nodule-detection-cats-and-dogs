from ultralytics import YOLO
import os

def main():
    model = YOLO('yolo26m.pt')

    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project='models',
        name='nodule_detection',
        exist_ok=True,
        save=True,
        plots=True,
        optimizer='MuSGD'
    )

if __name__ == '__main__':
    main()