import os
import glob
import pandas as pd

from sklearn.utils import shuffle


def get_files(root):
    lst = []
    for filepath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            lst.append(os.path.join(filepath,filename))
            
    return lst


def img_path_to_label_path(x, color=True):
    # z:\Data\dataset\apollo_lane_seg\ColorImage_road02\ColorImage \Record001\Camera 5\170927_063811892_Camera_5.jpg
    # z:\Data\dataset\apollo_lane_seg\          Labels_road02\Label\Record001\Camera 5\170927_063811892_Camera_5_bin.png
    # z:\Data\dataset\apollo_lane_seg\Gray_Label\Label_road02\Label\Record001\Camera 5\170927_063811892_Camera_5_bin.png
    if color:
        tmp = x.replace('ColorImage_', 'Labels_').replace("ColorImage", 'Label').replace(".jpg", '_bin.png') 
    else:
        tmp = x.replace('ColorImage_', 'Gray_Label/Label_').replace("ColorImage", 'Label').replace(".jpg", '_bin.png') 
    
    if not os.path.exists(tmp):
        return None
   
    return tmp
    

def get_train_val(dataset_path, color=True):
    files = []
    for i in glob.glob(dataset_path+"/ColorImage_road*"):
        # road 3存在几乎一半以上图像过曝的问题
        if "ColorImage_road03" in i:
            continue
        if 'tar' in i:
            continue 
        sub_folder = os.path.join(i, "ColorImage")
        # print(sub_folder)
        for i in get_files(sub_folder):
            files.append(i)

    df = pd.DataFrame(files, columns=['img'])
    df.loc[:, 'label'] = df.img.apply( lambda x: img_path_to_label_path(x, color))
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    df = shuffle(df, random_state=42)
    size = int(df.shape[0] * .8)
    
    print('get_train_val Done')
    
    return df[:size], df[size:]


if __name__ == '__main__':
    dataset_path = "/Data/dataset/apollo_lane_seg"
    # FIXME 灰度图不匹配
    '/Data/dataset/apollo_lane_seg/Gray_Label/Label_road04/Label/Record001/Camera 6/171206_054113660_Camera_6_bin.png'
    '/Data/dataset/apollo_lane_seg/ColorImage_road04/ColorImage/Record001/Camera 6/171206_054113660_Camera_6.jpg'
    df_train, df_val = get_train_val(dataset_path, color=True)
    
    df_train.to_csv('../data/apollo/data_list/train.csv', index=False)
    df_val.to_csv(  '../data/apollo/data_list/val.csv', index=False)

