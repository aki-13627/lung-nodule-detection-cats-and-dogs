from PIL import Image
import matplotlib.pyplot as plt

img_yolo = Image.open('images/grad_res_figures/figure-1.png')
img_rtdetr = Image.open('images/grad_res_figures/figure-2.png')


W, H = img_yolo.size


crop_width = int(W * 3 / 5)
crop1 = img_yolo.crop((0, 0, crop_width, H)) 
crop2 = img_rtdetr.crop((0, 0, crop_width, H)) 



fig, axes = plt.subplots(1, 2, figsize=(18, 10))
plt.subplots_adjust(wspace=0.05, hspace=0)


axes[0].imshow(crop1)
axes[0].axis('off')
axes[0].set_title('YOLO26', fontsize=20, fontweight='bold', pad=15) # 上に全体タイトルを追加


axes[1].imshow(crop2)
axes[1].axis('off')
axes[1].set_title('RT-DETR', fontsize=20, fontweight='bold', pad=15) # 上に全体タイトルを追加




fig.savefig('Figure_1_Combined_Loss_Panels.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

print("Figure_1_Combined_Loss_Panels.png を保存しました。")