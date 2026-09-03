from ultralytics import RTDETR
import pandas as pd

def main():
    img_sizes = [512, 640, 768, 896, 1024]
    
    
    results_summary = []

    for size in img_sizes:
        print(f"========== 解像度 {size} の検証を開始 ==========")
        
    
        model = RTDETR("rtdetr-l.pt")
        
    
        current_batch = 2 if size >= 896 else 4
        
    
        results = model.train(
            data="data.yaml",
            epochs=30,
            imgsz=size,
            batch=current_batch,
            device=0,
            project="outputs_rtdetr_imgsz_test",
            name=f"run_imgsz_{size}"
        )
        
    
        metrics = results.results_dict
        results_summary.append({
            "imgsz": size,
            "mAP50": metrics["metrics/mAP50(B)"],
            "Recall": metrics["metrics/recall(B)"],
            "Precision": metrics["metrics/precision(B)"]
        })

    df = pd.DataFrame(results_summary)
    df.to_csv("outputs_rtdetr_imgsz_test/imgsz_comparison.csv", index=False)
    print("\n========== 全サイズの比較が完了しました ==========")
    print(df)

if __name__ == "__main__":
    main()