import json
import os
import re

def merge_coco(base_json_path, new_json_path, output_json_path):
    with open(base_json_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    with open(new_json_path, 'r', encoding='utf-8') as f:
        new_data = json.load(f)

    # 既存の最大IDを取得
    max_img_id = max([img['id'] for img in base_data.get('images', [])] + [0])
    max_ann_id = max([ann['id'] for ann in base_data.get('annotations', [])] + [0])

    # カテゴリIDのマッピング（ベース側はID:1、追加側はID:0になっているため名前で紐付け）
    cat_map = {}
    base_cats = {cat['name'].replace('\\/', '/'): cat['id'] for cat in base_data.get('categories', [])}
    for cat in new_data.get('categories', []):
        cat_name = cat['name'].replace('\\/', '/')
        if cat_name in base_cats:
            cat_map[cat['id']] = base_cats[cat_name]
        else:
            cat_map[cat['id']] = base_data['categories'][0]['id']

    img_id_map = {}
    
    # 画像の統合とファイル名のクリーニング
    for img in new_data.get('images', []):
        old_id = img['id']
        max_img_id += 1
        new_id = max_img_id
        img_id_map[old_id] = new_id
        
        # パスを取り除き、先頭の「8文字の英数字-」を削除
        raw_name = img['file_name'].replace('\\/', '/')
        basename = os.path.basename(raw_name)
        clean_name = re.sub(r'^[a-fA-F0-9]{8}-', '', basename)
        
        new_img = img.copy()
        new_img['id'] = new_id
        new_img['file_name'] = clean_name
        base_data['images'].append(new_img)

    # アノテーションの統合とIDの更新
    for ann in new_data.get('annotations', []):
        if ann['image_id'] not in img_id_map:
            continue
            
        max_ann_id += 1
        new_ann = ann.copy()
        new_ann['id'] = max_ann_id
        new_ann['image_id'] = img_id_map[ann['image_id']]
        new_ann['category_id'] = cat_map.get(ann['category_id'], ann['category_id'])
        
        base_data['annotations'].append(new_ann)

    # 統合結果の保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(base_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # ファイルパスを適宜修正して実行してください
    BASE_JSON = "./coco_annotations.json"
    NEW_JSON = "./result.json"
    OUTPUT_JSON = "./merged_coco_annotations.json"
    
    merge_coco(BASE_JSON, NEW_JSON, OUTPUT_JSON)