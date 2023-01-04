#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector

#%%

class Detector():
    def __init__(self, config_file, checkpoint_file, device='cuda:0'):
        self.model = init_detector(config_file, checkpoint_file, device=device)
    
    def detect(self, img:Path, plot=False, out_file=None, *args, **kwargs):
        result = inference_detector(self.model, img)
        
        if out_file is not None and sum([len(lst) for lst in result]) != 0:
            self.model.show_result(img, result, out_file=out_file, *args, **kwargs)
        
        if plot:
            fig = plt.figure(figsize=(16,10))
            plt.imshow(cv2.cvtColor(self.model.show_result(img, result), cv2.COLOR_BGR2RGB))
        
        return result


config_file     = '../config/faster_rcnn.py'
checkpoint_file = '../work_dirs/faster_rcnn/latest.pth'

detector = Detector(config_file, checkpoint_file)

img = Path('./09005700121709091548010359Y_99.jpeg')
res = detector.detect(img, True, out_file="test.jpg", score_thr=0.5, thickness=1, font_size=10,)


img = Path('./09005700122003171015182885O_0.jpg')
res = detector.detect(img, True, out_file="test.jpg", score_thr=0.5, thickness=1, font_size=10,)


# %%
folder = Path('../data/panos/0000_0900570012_200322')
img_paths = list(folder.glob("*.jpg"))

for img in tqdm(img_paths):
    detector.detect(img, out_file=f"../cache/{img.name}", score_thr=0.5, thickness=1, font_size=10,)

# %%

