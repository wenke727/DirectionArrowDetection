# %%
import os
import cv2
import gc
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import geopandas as gpd 

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

from utils.parallel_helper import parallel_process

#%%
# TODO move to single python file
# TODO 将这些封装成类

def segmentation_to_geom(x):
    if len(x) == 1:
        geom = Polygon(np.array(x[0]).reshape(-1, 2)) 
    else: 
        geom = MultiPolygon( [Polygon(np.array(i).reshape(-1, 2)) for i in x if len(i) >= 6])
    
    return geom


def extract_info_from_bbox(bbox):
    x, y, w, h = bbox
    x_ = x + w/2
    y_ = y + h /2
    
    return {"x_center": x_, "y_center":y_, 'heigth': h, 'widht': w}


# %%


def query_classes_detail(coco):
    classes = [ val['name'] for _, val in coco.cats.items()]
    classes_num = len(classes)
    print(f"{len(classes)}, classes")    
    
    res = []
    for cat_ in classes:
        #采用getCatIds函数获取类别对应的ID
        ids = coco.getCatIds(cat_)
        #获取某一类的所有图片，比如获取包含dog的所有图片    
        imgIds = coco.catToImgs[ids[-1]]
        annIds = coco.getAnnIds(catIds=ids[-1])
        length = len(imgIds)
        info = {
            "id": ids[-1],
            "ids": ids,
            "cat": cat_,
            "length": length,
            "imgIds": imgIds,
            "annIds":annIds
        }
        res.append(info)
    
    return pd.DataFrame(res)


def extract_arrow(img_path, bbox, plot=True, format="RGB"):
    """提取箭头的区域

    Args:
        img_path ([type]): [description]
        bbox ([type]): [description]
        plot (bool, optional): [description]. Defaults to True.
        format (str, optional): [description]. Defaults to "RGB".

    Returns:
        [type]: [description]
    """
    im = cv2.imread(img_path)
    x, y, w, h = bbox
    crop_img = im[y:y+h, x:x+w]

    if plot:
        plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

    if format == 'RGB':
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    
    return crop_img


def coco_extract_arrow(coco, imgId, annId, verbose=False, plot=False, format="RGB"):
    """提取要素
    """
    img_info = coco.loadImgs(imgId)[0]
    ann_info = coco.loadAnns(annId)[0]

    if verbose:
        print(img_info)
    # extract_arrow(imPath, anns[1]['bbox'])
    img = extract_arrow(img_info['file_name'], ann_info['bbox'], plot, format)
    
    return img


def coco_show_img_anns(coco, imgId):
    """展示一个图片中所有的label

    Args:
        coco ([type]): [description]
        imgId ([type]): [description]
    """
    img_dict = coco.loadImgs(ids=imgId)[0]
    anns_lst = coco.loadAnns(coco.getAnnIds(imgIds=[imgId]))

    plt.figure(figsize=(16, 10))
    fig = plt.imshow(Image.open(img_dict['file_name']))
    coco.showAnns(anns_lst, draw_bbox=True)
    
    return 


def get_ann_df(annIds):

    ann_dict = {}
    for annId in annIds:
        ann_dict[annId] = coco.loadAnns(annId)[0]
        
    df = pd.DataFrame(ann_dict).T
    df.loc[:, 'width'] = df.bbox.apply(lambda x: x[2])
    df.loc[:, 'height'] = df.bbox.apply(lambda x: x[3])
    
    return df


def plot_arrows(df, cat_id, startPage=0):
    """
    绘制九宫格
    """    
    imgIds, annIds = df.iloc[cat_id].imgIds, df.iloc[cat_id].annIds

    nrows, ncols = 10, 5
    plt.figure(figsize=(5 * ncols, 5 * nrows))

    start = nrows * ncols * startPage
    end = start + nrows * ncols if  start + nrows * ncols < len(imgIds) else len(imgIds) 
    for i in tqdm(range(start, end )):
        img = coco_extract_arrow(coco, imgIds[i], annIds[i], )
        std_ = np.mean([np.std(img[:,:, i]) for i in range(3)])

        ax = plt.subplot(nrows, ncols, i + 1 - start)
        plt.imshow(img)
        ax.set_title(f"{annIds[i]} / {imgIds[i]} ({std_:.0f})")
    
    return


def extract_arrows(df, cat_id, folder='../cache'):
    """extract the arrows area, and write it to folder

    Args:
        df ([type]): [description]
        cat_id ([type]): [description]
        folder (str, optional): [description]. Defaults to '../cache'.
    """
    folder = os.path.join(folder, coco.loadCats([cat_id])[0]['name'])
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    imgIds, annIds = df.iloc[cat_id].imgIds, df.iloc[cat_id].annIds

    # for imgID, annID in tqdm(zip(list(imgIds), list(annIds))[:100]):
    for i in tqdm(range(len(imgIds))):
        imgID, annID = imgIds[i], annIds[i]
        img = coco_extract_arrow(coco, imgIds[i], annIds[i], )
        fn = os.path.join(folder, f"{annID}_{imgID}.jpg")
        cv2.imwrite(fn, img)
        
    return


def vis_label_and_img(coco, imgId, annID, folder='../cache'):
    fig = plt.figure(figsize=(16,10))
    ax1 = plt.axes([0,0,1,1])

    # the whole image
    gdf_ann.loc[[annID]].plot(edgecolor='r', facecolor='none', linestyle='--', alpha=.9, legend=True, ax=ax1)
    background = cv2.imread(coco.loadImgs(ids=[imgId])[0]['file_name'])
    background[:700,:] = [255,255,255]
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))

    _, w = ax1.get_xlim()
    plt.hlines(800, 0, w, color='white',linestyles=':', alpha=.6)
    plt.hlines(1000, 0, w, color='white',linestyles=':', alpha=.6)
    plt.axis('off')

    # partial enlarged detail
    ax2 = plt.axes([0.35, 0.6, .3, .2])
    img = coco_extract_arrow(coco, imgId, annID, plot=False)
    plt.imshow(img)
    plt.axis('off')
    item = gdf_ann.loc[annID]
    ax2.set_title(f"{item.name}/{imgId}, {item.bbox[1]}, {item.area}{item.bbox[2:]}")

    if folder is not None:
        fn = os.path.join(folder, f"{annID:05d}_{imgId:05d}.jpg")
        plt.savefig(fn, bbox_inches='tight', pad_inches=0, dpi=120)

    plt.close()
    
    return fig


def vis_arrow_img_helper(item, coco):
    return vis_label_and_img(coco, int(item.image_id), int(item.id))


# %%
# load coco dataset
cocoRoot = '/Data/repos/mmdetection/data/apollo'
dataType = 'train'
file_name = "train.json"
label_file = os.path.join(cocoRoot, f"annotations/{file_name.split('.')[0]}.json")

label_file = '../src/test.json'
label_file = '/Data/repos/traffic_sign_detection/data/apollo/dir_arrow_val_1216.json'

coco = COCO(label_file)
df_cat_details = query_classes_detail(coco).query('length > 0')


cat_id = 1
imgIds, annIds = df_cat_details.iloc[cat_id].imgIds, df_cat_details.iloc[cat_id].annIds

gdf_ann = gpd.GeoDataFrame(coco.loadAnns(coco.getAnnIds()))
gdf_ann.loc[:, 'geometry'] = gdf_ann.segmentation.apply(segmentation_to_geom)
gdf_ann.index = gdf_ann.id
# annDf[['x_center', 'y_center', 'height', 'widht']] = annDf.apply(lambda x: extract_info_from_bbox(x.bbox), axis=1, result_type='expand')

print(f"Total label arrows: {df_cat_details.length.sum()}")

i = 1
df_cat_details

# coco_show_img_anns(coco=coco, imgId=4)
# plot_arrows(df_cat_details, 1, 1)
# extract_arrows(df_cat_details, 3)



#%%


# ! extract_arrow 绘制单独的箭头，并划分区域
def vis_arrow_img_helper(item, coco=coco):
    return vis_label_and_img(coco, int(item.image_id), int(item.id), item.folder)


def export_labels_for_checking(gdf_ann, cat_ids=None, folder='../cache'):
    if cat_ids is None:
        cat_ids = list(gdf_ann.category_id.unique())
    if type(cat_ids) != list:
        cat_ids = [cat_ids]
    
    for cat_id in cat_ids:
        root_folder='../cache'
        folder = os.path.join(root_folder, coco.loadCats([cat_id])[0]['name'])
        if not os.path.exists(folder):
            os.makedirs(folder)

        df = gdf_ann.query(f'category_id == {cat_id}')
        df.loc[:, 'folder'] = folder

        figs = parallel_process(df, vis_arrow_img_helper, n_jobs=32)
        del figs
        gc.collect()
    
    return

# Serial operation
# id = 24
# imgId, annID = imgIds[id], annIds[id]
# for id in tqdm(range(len(imgIds))):
#     fig = vis_arrow_img(coco, imgIds[id], annIds[id])



export_labels_for_checking(gdf_ann, cat_ids=None)

#%%


# %%

