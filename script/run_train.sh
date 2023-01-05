CONFIG=./config/faster_rcnn.py

CUDA_VISIBLE_DEVICES=7 python src/train.py $CONFIG --gpus 1 --work-dir results
