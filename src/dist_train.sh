#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,5

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

echo "$(dirname $0)/.."

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
echo $PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
