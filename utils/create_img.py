import os
from PIL import Image

def extract_individual_metrics_without_title():
    out_dir = "images/metrices"
    os.makedirs(out_dir, exist_ok=True)
    
    img_yolo = Image.open('images/grad_res_figures/figure-1.png')
    img_rtdetr = Image.open('images/grad_res_figures/figure-2.png')
    
    W, H = img_yolo.size
    w = W // 5
    h = H // 2
    
    crop_top_pixels_row0 = 66
    crop_top_pixels_row1 = 50
    
    def get_crop_no_title(img, col, row):
        left = col * w
        
        if row == 0:
            upper = (row * h) + crop_top_pixels_row0
        else:
            upper = (row * h) + crop_top_pixels_row1
            
        right = (col + 1) * w
        lower = (row + 1) * h
        return img.crop((left, upper, right, lower))

    tasks = [
        (img_yolo,   3, 0, "YOLO26_Precision.png"),
        (img_rtdetr, 3, 0, "RT-DETR_Precision.png"),
        (img_yolo,   4, 0, "YOLO26_Recall.png"),
        (img_rtdetr, 4, 0, "RT-DETR_Recall.png"),
        (img_yolo,   3, 1, "YOLO26_mAP50.png"),
        (img_rtdetr, 3, 1, "RT-DETR_mAP50.png"),
        (img_yolo,   4, 1, "YOLO26_mAP50-95.png"),
        (img_rtdetr, 4, 1, "RT-DETR_mAP50-95.png"),
    ]
    
    for img, col, row, filename in tasks:
        cropped_img = get_crop_no_title(img, col, row)
        save_path = os.path.join(out_dir, filename)
        cropped_img.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    extract_individual_metrics_without_title()