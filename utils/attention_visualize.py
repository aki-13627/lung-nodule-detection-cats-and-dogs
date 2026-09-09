import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import RTDETR

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = RTDETR("./runs/detect/outputs_rtdetr/lung_nodule_recall_run-23/weights/best.pt")

target_layer = None
for name, module in model.model.named_modules():
    if "AIFI" in type(module).__name__:
        target_layer = module
        break

if target_layer is not None:
    target_layer.register_forward_hook(get_activation("feature_map"))

img_path = "yolo_dataset/images/train/00061496_suzuki_cocoa_01.png"

img = cv2.imread(img_path)
img = cv2.resize(img, (512, 512))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model.predict(img_rgb, imgsz=512, conf=0.80, iou=0.4, device="mps")

boxes = results[0].boxes
if len(boxes) > 0:
    box = boxes[0].xyxy[0].cpu().numpy()
    target_x = int((box[0] + box[2]) / 2)
    target_y = int((box[1] + box[3]) / 2)
    print(f"Detected Bbox Center: x={target_x}, y={target_y}")
else:
    print("Warning: Default coordinates used.")
    target_x = 256
    target_y = 256

if "feature_map" in activation:
    features = activation["feature_map"]

    if features.dim() == 3:
        b, seq_len, embed_dim = features.shape
        size = int(np.sqrt(seq_len))
        features = features.transpose(1, 2).reshape(b, embed_dim, size, size)
    elif features.dim() == 4:
        pass 

    features = features.squeeze(0)
    _, h, w = features.shape

    feature_x = int((target_x / 512) * w)
    feature_y = int((target_y / 512) * h)

    target_vector = features[:, feature_y, feature_x].unsqueeze(1).unsqueeze(2)
    
    similarity = F.cosine_similarity(features.unsqueeze(0), target_vector.unsqueeze(0), dim=1).squeeze().cpu().numpy()

    sim_min = np.min(similarity)
    sim_max = np.max(similarity)
    similarity = (similarity - sim_min) / (sim_max - sim_min + 1e-8)

    similarity = (similarity) ** 2000

    heatmap_resized = cv2.resize(similarity, (512, 512))

    yellow_heatmap = np.zeros((512, 512, 4), dtype=np.float32)
    yellow_heatmap[..., 0] = 1.0  
    yellow_heatmap[..., 1] = 1.0  
    yellow_heatmap[..., 2] = 0.0  
    yellow_heatmap[..., 3] = heatmap_resized * 0.8

    half_size = 75
    ymin = max(0, target_y - half_size + 30)
    ymax = min(512, target_y + half_size)
    xmin = max(0, target_x - half_size)
    xmax = min(512, target_x + half_size)

    mask = np.zeros((512, 512), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    yellow_heatmap[~mask, 3] = 0.0

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.scatter(target_x, target_y, color='red', marker='x', s=100)
    plt.axis("off")
    plt.savefig("original_image.jpg", bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Saved original_image.jpg")

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.imshow(yellow_heatmap)
    plt.axis("off")
    plt.savefig("pixel_attention_overlay.jpg", bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Saved pixel_attention_overlay.jpg")