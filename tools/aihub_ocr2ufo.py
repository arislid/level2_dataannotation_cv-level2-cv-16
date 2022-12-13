import os
from pathlib import Path
import json
import shutil

label_path = "./공공행정문서 OCR/[라벨]train"
mk_path = "./SimpleOCR_atom/images"
image_path = "./공공행정문서 OCR/[원천]train"
level2_path = os.listdir(label_path)


class label:
    def __init__(self, data):
        self.info = data["images"][0]
        self.annos = data["annotations"]
        sorted(self.annos, key=lambda x : x["id"])
        
        self.bbox = []
        self.value = []
        for anno in self.annos:
            if anno["annotation.type"] != 'rectangle':
                print(anno["annotation.type"])
            self.bbox.append(self.change_bbox(anno["annotation.bbox"]))
            self.value.append(anno["annotation.text"])
      
        self.file_name = self.info["image.file.name"]
        self.width = self.info["image.width"]
        self.height = self.info["image.height"]
    
    def name(self):
        return self.file_name
    
    def size(self):
        return (self.width, self.height)
    
    def change_bbox(self, t_bbox):
        x, y, w, h = t_bbox
        return [(x, y), (x+w, y), (x+w,y+h), (x, y+h)]
    
    def __getitem__(self, index):
        return self.bbox[index], self.value[index]
    
    def __len__(self):
        return len(self.bbox)

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def mkdir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print ('Error: Creating directory. ' +  dir)
        
new_data = {}
new_data["images"] = {}
mkdir(mk_path)
for level2 in level2_path:
    level3_path = os.listdir(os.path.join(label_path, level2))
    for level3 in level3_path:
        level4_path = os.listdir(os.path.join(label_path, level2, level3))
        for level4 in level4_path:
            file_path = os.listdir(os.path.join(label_path, level2, level3, level4))
            count = 0
            for file_name in file_path:
                data = read_json(os.path.join(label_path, level2, level3, level4, file_name))
                t_label = label(data)
                img_path = os.path.join(image_path, level2, level3, level4, t_label.name())
                if not os.path.exists(img_path):
                    print(f"Img not exist : {img_path}")
                    continue
                shutil.copyfile(img_path, os.path.join(mk_path, t_label.name()))
                
                
                word_format = {}
                for idx, (bbox, value) in enumerate(t_label):
                    str_idx = idx
                    word_format[str_idx] = {
                            "transcription" : value,
                            "points":bbox,
                            "orientation":"Horizontal",
                            "language":["ko"],
                            "tags":[],
                            "confidence": None,
                            "illegibility": False
                        }
                
                ufo_format = {
                    "paragraphs":{},
                    "words": word_format,
                    "chars": {},
                    "img_w": t_label.width,
                    "img_h": t_label.height,
                    "tags": [],
                    "relations": {},
                    "annotation_log": {
                        "worker": "AI_hub",
                        "timestamp": "2022-12-07",
                        "tool_version": "",
                        "source": None
                        }
                    ,
                    "license_tag": {
                        "usability": True,
                        "public": False,
                        "commercial": True,
                        "type": None,
                        "holder": "Upstage"
                        }
                    
                }
                new_data["images"][t_label.name()] = ufo_format
                print(f"ADD ufo format : {t_label.name()}")
                if count > 5:
                    break
                count += 1
with open("./SimpleOCR_atom/ufo/fixed_train.json","w", encoding='utf-8') as outfile:
    json.dump(new_data, outfile, ensure_ascii=False)