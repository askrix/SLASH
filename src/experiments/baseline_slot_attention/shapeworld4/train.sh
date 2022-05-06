#!/bin/bash

#DATA="../../data/shapeworld4"
DATA="/workspace/datasets-local/data/shapeworld4"
MODEL="slot-attention-set-pred-shapeworld4"
DATASET=shapeworld4
DEVICE=$1
SEED=$2 # 0, 1, 2, 3, 4
CREDENTIALS=$3
#-------------------------------------------------------------------------------#
# Train on CLEVR_v1 with cnn model

CUDA_VISIBLE_DEVICES=$DEVICE python3 train.py \
--data-dir $DATA --dataset $DATASET --epochs 1100 \
--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed $SEED \
--warmup-epochs 8 --decay-epochs 368 --num-workers 0 --credentials $CREDENTIALS

#CUDA_VISIBLE_DEVICES=$DEVICE python3 src_concept_learner/slot_attention_set_prediction_shapeworld4/train.py \
#--data-dir $DATA --dataset $DATASET --full-eval --epochs 1500 \
#--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed 14 \
#--warm-up-epochs 0 --num-workers 0
#
#CUDA_VISIBLE_DEVICES=$DEVICE python3 src_concept_learner/slot_attention_set_prediction_shapeworld4/train.py \
#--data-dir $DATA --dataset $DATASET --full-eval --epochs 1500 \
#--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed 21 \
#--warm-up-epochs 0 --num-workers 0
#
#CUDA_VISIBLE_DEVICES=$DEVICE python3 src_concept_learner/slot_attention_set_prediction_shapeworld4/train.py \
#--data-dir $DATA --dataset $DATASET --full-eval --epochs 1500 \
#--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed 33 \
#--warm-up-epochs 0 --num-workers 0
#
#CUDA_VISIBLE_DEVICES=$DEVICE python3 src_concept_learner/slot_attention_set_prediction_shapeworld4/train.py \
#--data-dir $DATA --dataset $DATASET --full-eval --epochs 1500 \
#--name $MODEL --lr 0.0004 --batch-size 512 --n-slots 4 --n-iters-slot-att 3 --n-attr 15 --ap-log 1 --seed 104 \
#--warm-up-epochs 0 --num-workers 0

