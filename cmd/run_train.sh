# cascade_rcnn
# ps auxww | grep 'rcnn' | awk '{print $2}' | xargs kill -9

# Fabric
# CONFIG=./justForFun/cascade_rcnn_r50_fpn_1x_gd.py

# TT100K
# CONFIG=./justForFun/cascade_rcnn_r50_fpn_1x_TT100K.py

# mask_rcnn
CONFIG=./config/faster_rcnn.py


CUDA_VISIBLE_DEVICES=7 python src/train.py $CONFIG --gpus 1
