from ultralytics import RTDETR

def main():
    
    model_path = "runs/detect/outputs_rtdetr/lung_nodule_run-5/weights/best.pt"
    model = RTDETR(model_path)
    
    
    metrics = model.val(
        data="data.yaml",
        conf=0.001,  # 曲線を描くために、閾値を低めにして全範囲のデータを収集します
        iou=0.5,
        save_json=True, 
        plots=True 
    )
    
    print("評価が完了しました。保存されたグラフを確認してください。")

if __name__ == "__main__":
    main()