# %%

import os
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import box, LineString, Polygon, MultiPolygon


from utils.apollo_labels import labels, name2label, id2label, trainId2label, category2labels, get_color_to_label
from utils.mask_to_bbox import mask_to_bbox, mask_to_border
from utils.parallel_helper import parallel_process


parent_path = '../data/apollo'
DIRECTION_CAT = 7
color_to_label = get_color_to_label(labels, [DIRECTION_CAT], with_ignored=True)

category_instancesonly = [ val['name'] for _, val in color_to_label.items() ]

# %%

def plot_contours(contours, img_path, fig=None):
    """Draw contours

    Args:
        contours ([type]): [description]
        img_path ([type]): [description]
    """
    # if fig is None:
    plt.figure(figsize=(20, 15))
    plt.imshow( Image.open(img_path) )
    for i, c in enumerate(contours):
        plt.plot(c[:, 1], c[:, 0], linewidth=2, label=str(i))
        
    plt.legend()
    
    return 


def plot_contours_bboxes(contours, bboxes, img_path, fig=None):
    """Draw contours

    Args:
        contours ([type]): [description]
        img_path ([type]): [description]
    """
    # if fig is None:
    plt.figure(figsize=(16, 10))
    plt.imshow( Image.open(img_path) )
   
    for i, c in enumerate(contours):
        plt.plot(c[:, 1], c[:, 0], linewidth=2, label=str(i))
    
    for i, box in enumerate(bboxes):
        cat, x0, y0, x1, y1 = box
        lines = np.array([ [x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]])
        plt.plot(lines[:, 0], lines[:, 1], linewidth=2, label=cat)
        
    plt.legend()
        
    
    return 


def convert_img_to_gray(img_path, plot=False):
    lab_as_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
    w, h = lab_as_np.shape[0], lab_as_np.shape[1]
    is_gray = is_gray_img(lab_as_np)
    if is_gray:
        return lab_as_np[:,:,0][:, :, np.newaxis].astype(np.uint8)

    encode_mask = np.zeros((w, h))
    for color, val in color_to_label.items():
        mask = get_mask(lab_as_np, [val['id']] if is_gray else color)
        if np.sum(mask) == 0:
            continue
        encode_mask[mask] = val['id']
        
    if plot:
        plt.imshow(encode_mask)
    
    # return encode_mask[:, :, np.newaxis].astype(np.uint8)
    return encode_mask[:, :, np.newaxis].astype(np.uint8)


def get_mask(img, color=[100, 100, 150]):
    if len(color) == 1:
        return img[:,:,0] == color[0]
    
    return (img[:,:,0] == color[0]) & (img[:,:,1] == color[1]) & (img[:,:,2] == color[2])


def is_gray_img(img):
    a = np.sum(img[:,:,0] != img[:,:,1])
    if a != 0:
        return False
    
    b = np.sum(img[:,:,0] != img[:,:,2])
    if b != 0:
        return False
    
    c = np.sum(img[:,:,1] != img[:,:,2])
    if c != 0:
        return False
    
    return True


def combine_difficult_anno(gdf, img_fn=None, plot=None):
    gdf_ = gdf[gdf.maybe_split]
    drop_index, obj_lst_comb = [], []
    convert_to_geom = True

    # 合并相邻的地面标识；
    # TODO 是否需要增加更多的信息; 然后按照区域所在位置聚类
    for cat_, df in gdf_.groupby('category'):
        if df.shape[0] != 2 or cat_ == 'a_w_t':
            continue
        
        for i in df.index:
            drop_index.append(i)
        
        seg0, seg1 = df.iloc[0].segmetation, df.iloc[1].segmetation
        b0, b1 = df.iloc[0].bbox, df.iloc[1].bbox
        
        bbox = [ min(b0[0], b1[0]), min(b0[1], b1[1]), max(b0[2], b1[2]), max(b0[3], b1[3])]
        segmetation = seg0
        for seg in seg1:
            segmetation.append(seg)

        info = {'category': cat_, 'bbox': bbox, "segmetation":segmetation}
        if convert_to_geom:
            info['geometry'] = MultiPolygon( [Polygon(np.array(x).reshape(-1, 2)) for x in info['segmetation'] if len(x) >= 6 ] )
        obj_lst_comb.append(info)

    gdf_combine = gdf.append(obj_lst_comb, ignore_index=True).drop(index=drop_index)

    
    if plot:
        fig = plt.figure(figsize=(15, 3.5))
        cols = gdf_combine.shape[0]
        if img_fn is not None:
            img_file = Image.open(img_fn)

        for img_id, i in enumerate(gdf_combine.index):
            ax = plt.subplot(1, cols, img_id+1)
            ax = gdf_combine.loc[[i]].geometry.plot(ax=ax, edgecolor='r', facecolor='none', linestyle='--', alpha=.8)
            
            y0, y1 = ax.get_ylim()
            std_ = None
            if img_fn is not None:
                x0, x1 = ax.get_xlim()
                plt.imshow(img_file)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                # TODO add pixel std
                img_np = cv2.imread(img_fn)[int(x0):int(x1), int(y0):int(y1), :]
                std_ = np.mean([np.std(img_np[:, :, i]) for i in range(3)])
                
            ax.set_ylim(ax.get_ylim()[::-1])
            if std_ is None:
                ax.set_title(gdf_combine.loc[i].category)
            else:
                ax.set_title(f"{gdf_combine.loc[i].category}, {std_:.1f}")
        plt.tight_layout()

    return gdf_combine


def visulize_find_contours(img_, gdf):
    """针对提取箭头的可视化

    Args:
        img_ ([type]): [description]
        ans ([type]): [description]
    """

    ax = gdf.plot(column= 'category', alpha=.25, figsize=(20, 15), legend=True)
    gdf_ = gdf.copy()
    gdf_.loc[:, 'geometry'] = gdf_.bbox.apply(lambda x: box(*x), )
    gdf_.plot(ax=ax, edgecolor='r', facecolor='none', linestyle='--', alpha=.8)
    
    plt.imshow(img_)


def get_bbox(img_fn, label_fn, mark=True, combine=True, plot_arrows=True, verbose=False, thres=600):
    lab_as_np = cv2.imread(label_fn, cv2.IMREAD_COLOR)
    w, h = lab_as_np.shape[0], lab_as_np.shape[1]
    is_gray = is_gray_img(lab_as_np)

    # step 1: mask to bbox
    bboxes, contours_dict = {}, {}
    for color, val in color_to_label.items():
        encode_mask = np.zeros((w, h))
        mask = get_mask(lab_as_np, [val['id']] if is_gray else color)
        if np.sum(mask) == 0:
            continue
        
        encode_mask[mask] = val['id']
        # 通过区域大小筛选候选区域，框的大小暂定为 “500”
        contours_candidates, bbox = mask_to_bbox(encode_mask, area_thres=thres, height_thres=12)
        
        bboxes[val['name']] = bbox
        contours_dict[val['name']] = contours_candidates

        if verbose:
            print(f"{val['name']}: bbox: {len(bbox)}, contours: {len(contours_candidates)}")
    
    # step 2: bbox to obj_list
    obj_list = []
    for key, lst in bboxes.items():
        for i in lst:
            geom = box(*i)
            contours_geoms = [LineString([[i[1], i[0]] for i in line ]) for line in contours_dict[key]]
            
            lst, coords_lst = [], []
            for _, item in enumerate(contours_geoms):
                if item.intersects(geom):
                    coords = np.array(item.simplify(1).coords).astype(np.int)
                    lst.append(coords.reshape(-1).tolist())
                    coords_lst.append(coords)
            obj_list.append({'category': key, 'bbox': i, "segmetation":lst})
    if len(obj_list) == 0 or (len(obj_list) == 1 and obj_list[0]['category'] == 'void' ):
        if verbose:
            print("Negative sample")
        return None, None

    # step 3: obj combine
    gdf_combine = None
    if combine:
        gdf = extract_difficult_anno(obj_list)
        gdf_combine = combine_difficult_anno(gdf, img_fn, plot_arrows)
        obj_list = [val for key, val in gdf_combine[['category', 'bbox', 'segmetation']].to_dict('index').items()]

    if mark:
        # FIXME: update
        """ marking bounding box on image """
        visulize_find_contours(Image.open(img_fn), gdf_combine)
    
    return obj_list, gdf_combine


def extract_difficult_anno(obj_list):
    gdf = gpd.GeoDataFrame(obj_list)
    gdf.loc[:, 'geometry'] = gdf.segmetation.apply(lambda x: 
        Polygon(np.array(x[0]).reshape(-1, 2)) 
        if len(x) == 1 
        else 
            MultiPolygon( [Polygon(np.array(i).reshape(-1, 2)) for i in x if len(i) >= 6])
    )
    gdf_void = gdf.query("category == 'void' ")
    gdf.query("category != 'void' ", inplace=True)

    # see: https://stackoverflow.com/questions/63955752/topologicalerror-the-operation-geosintersection-r-could-not-be-performed
    gdf.loc[:, 'maybe_split'] = gdf.buffer(0).intersects(gdf_void.iloc[0].geometry.buffer(2))
    gdf.loc[:, 'area_'] = gdf.geometry.area
    gdf.loc[:, 'width_'] = gdf.bbox.apply(lambda x: x[2]-x[0])
    gdf.loc[:, 'height_'] = gdf.bbox.apply(lambda x: x[3]-x[1])

    gdf.loc[:, 'aspect_ratio'] = gdf.width_ / gdf.height_
    
    gdf.loc[:, 'x_c'] = gdf.bbox.apply(lambda x: (x[2]+x[0])//2)
    gdf.loc[:, 'y_c'] = gdf.bbox.apply(lambda x: (x[3]+x[1])//2)
    
    return gdf


def extract_img_labels(item, plot=False):
    image_id = item.name

    image_path, label_path = item['img'], item['label']

    im = cv2.imread(image_path)
    H, W, _ = im.shape

    # 添加图像的信息到dataset中
    img_info = {
        'file_name': image_path, 
        'label_name': label_path,
        'id': image_id,
        'width': W,
        'height': H
    }

    obj_list, _ = get_bbox(image_path, label_path, mark=plot, plot_arrows=False)

    return {'img_info': img_info, "obj_list": obj_list}


def dataset_label_initial(parent_path):
    os.makedirs(os.path.join(parent_path, 'annotations'), exist_ok=True)

    dataset = {'info':{},'licenses':[],'categories': [],'images': [] , 'annotations': []}
    label = {}

    info = {
        "year": 2021, # 年份
        "version": '1.0',# 版本
        "description": "Apollo lane segmetation", # 数据集描述
        "contributor":"Baidu",# 提供者
        "url":'https://aistudio.baidu.com/aistudio/competition/detail/5/0/introduction',# 下载地址
        "date_created":2021-1-15
    }

    licenses = {
        "id" :1,
        "name" :"null",
        "url" :"null",
    }

    dataset['info']= info
    dataset['licenses'] = licenses

    #建立类别和id的关系
    for i, item in enumerate(color_to_label.values()):
        dataset['categories'].append({'id': i, 'name': item['name'], 'supercategory': item['catId']})
        label[item['name']]=i

    return dataset, label


def add_obj_list_to_dataset(dataset, obj_list, image_id, obj_id, label):
    if obj_list == None:
        return obj_id
    
    for anno_dic in obj_list:
        label_key = anno_dic['category']
        x, y, xmax, ymax = anno_dic['bbox']
        width = xmax - x
        height = ymax - y
        area = width * height
       
        dataset['annotations'].append({
                                    'area': area,
                                    'bbox': [x, y, width, height],
                                    'category_id':label[label_key],
                                    'id': obj_id,
                                    'image_id': image_id,
                                    'iscrowd': 0,
                                    # mask, 矩形是从左上角点按顺时针的四个顶点
                                    # 'segmentation': [[x, y, x+width, y, x+width, y+height, x, y+height]]
                                    'segmentation': anno_dic['segmetation']
                                })
        #每个标注的对象id唯一
        # print(f"obj_id: {obj_id}")
        obj_id += 1
    
    return obj_id


def check_single():
    root_folder = "/Data/dataset/apollo_lane_seg"

    img_fn =  os.path.join(root_folder,        'ColorImage_road02/ColorImage/Record007/Camera 5/170927_064610055_Camera_5.jpg')
    label_fn  = os.path.join(root_folder,               "Labels_road02/Label/Record007/Camera 5/170927_064610055_Camera_5_bin.png")
    gray_label_fn = os.path.join(root_folder, "Gray_Label/Label_road02/Label/Record007/Camera 5/170927_064610055_Camera_5_bin.png")


    obj_list, gdf_combine = get_bbox(img_fn, gray_label_fn, True)


def df_check_single(gdf, id):
    item = gdf.iloc[id]
    image_path, label_path = item['img'], item['label']
    obj_list, _ = get_bbox(image_path, label_path, True, plot_arrows=True)
    
    return


def run(input, output, n_jobs=32):
    dataset, label = dataset_label_initial(parent_path)
    gdf = pd.read_csv(input)
    res = parallel_process(gdf, extract_img_labels, n_jobs=n_jobs)

    obj_id = 1
    for item in res:
        dataset['images'].append(item['img_info'])
        obj_id = add_obj_list_to_dataset(dataset, item['obj_list'], item['img_info']['id'], obj_id, label)

    with open(output, 'w') as fn:
        fn.write(json.dumps(dataset, indent=1))
    
    

# %%
if __name__ == "__main__":
    run(input='../data/apollo/data_list/val.csv', output='../data/apollo/annotations/dir_arrow_val_1216.json', n_jobs=50)
    # run(input='../data/apollo/data_list/train.csv', output='../data/apollo/annotations/dir_arrow_train_1216.json', n_jobs=50)
    
    # df_check_single(gdf, 12)
    pass

#%%