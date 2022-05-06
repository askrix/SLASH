#!/bin/bash

DATA="../../experiments/data/shapeworld4"
MODEL="slot-attention-set-pred-shapeworld4"
DATASET=shapeworld4
DEVICE=0
#SEED=$2 # 0, 14, 21, 33, 104
#-------------------------------------------------------------------------------#
# Train on CLEVR_v1 with cnn model

#CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#--data-dir $DATA --dataset $DATASET --full-eval --epochs 1500 \
#--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed 0 \
#--warm-up-epochs 0 --decay-epochs 1000 --num-workers 0
#
#CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#--data-dir $DATA --dataset $DATASET --full-eval --epochs 1500 \
#--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed 1 \
#--warm-up-epochs 0 --decay-epochs 1000 --num-workers 0

#CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#--data-dir $DATA --dataset $DATASET --full-eval --epochs 1500 \
#--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed 2 \
#--warm-up-epochs 0 --decay-epochs 1000 --num-workers 0

#CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
#--data-dir $DATA --dataset $DATASET --full-eval --epochs 1500 \
#--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed 3 \
#--warm-up-epochs 0 --decay-epochs 1000 --num-workers 0

CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
--data-dir $DATA --dataset $DATASET --full-eval --epochs 1000 \
--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed 4 \
--warm-up-epochs 8 --decay-epochs 360 --num-workers 8
