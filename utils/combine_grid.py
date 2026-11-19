import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_yolo = mpimg.imread('images/pred_new/val_predictions_yolo_grid_2_replaced.png')
img_rtdetr = mpimg.imread('images/pred_new/val_predictions_rtdetr_grid_2.png')

fig, axes = plt.subplots(1, 2, figsize=(16, 8))


axes[0].imshow(img_yolo)
axes[0].axis('off')

axes[0].text(0.0, 1.02, '(A)', transform=axes[0].transAxes, 
             fontsize=18, va='bottom', ha='left', fontweight='bold')

axes[0].text(0.5, 1.02, 'YOLO26', transform=axes[0].transAxes, 
             fontsize=18, va='bottom', ha='center', fontweight='bold')


axes[1].imshow(img_rtdetr)
axes[1].axis('off')

axes[1].text(0.0, 1.02, '(B)', transform=axes[1].transAxes, 
             fontsize=18, va='bottom', ha='left', fontweight='bold')

axes[1].text(0.5, 1.02, 'RT-DETR', transform=axes[1].transAxes, 
             fontsize=18, va='bottom', ha='center', fontweight='bold')

plt.tight_layout()

output_filename = 'combined_figure_3_new.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

print(f"画像を結合し、{output_filename} として保存しました。")