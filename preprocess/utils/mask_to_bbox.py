import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours, approximate_polygon


def create_dir(path):
    """ Creating a directory """
    if not os.path.exists(path):
        os.makedirs(path)


def mask_to_border(mask, thres=1):
    """ Convert a mask to border image """
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, thres)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return contours, border


def mask_to_bbox(mask, area_thres=0, height_thres=20):
    """ Mask to bounding boxes """
    bboxes = []

    contours, mask = mask_to_border(mask)
    # label函数即可实现连通区域标记
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        if (x2-x1)*(y2-y1) > area_thres and y2-y1 > height_thres:
            bboxes.append([x1, y1, x2, y2])

    return contours, bboxes


def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


if __name__ == "__main__":
    """ Load the dataset """
    images = sorted(glob(os.path.join("data", "image", "*")))
    masks = sorted(glob(os.path.join("data", "mask", "*")))

    """ Create folder to save images """
    create_dir("results")

    """ Loop over the dataset """
    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Read image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """ Detecting bounding boxes """
        bboxes = mask_to_bbox(y)

        """ marking bounding box on image """
        for bbox in bboxes:
            x = cv2.rectangle(x, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        """ Saving the image """
        cat_image = np.concatenate([x, parse_mask(y)], axis=1)
        cv2.imwrite(f"results/{name}.png", cat_image)