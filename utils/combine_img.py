from PIL import Image

img1 = Image.open('./images/patient_level_cm_yolo26.png')
img2 = Image.open('./images/patient_level_cm_rtdetr.png')

total_width = img1.width + img2.width
max_height = max(img1.height, img2.height)

dst = Image.new('RGB', (total_width, max_height), (255, 255, 255))

dst.paste(img1, (0, 0))
dst.paste(img2, (img1.width, 0))


dst.save('combined_cm.png')