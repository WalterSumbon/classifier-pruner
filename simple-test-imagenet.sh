#!/bin/bash
#SBATCH -N 1 -n 8 -p GPU-V100 --gres=gpu:v100:1
module load cuda/10.2.89
module load anaconda3
module load python/3.8.1
module load cudnn/7.6.5/cuda/9.2
module load openmpi/4.0.3/gcc/9.3.0
source activate ns

device=0
prefix=exp_slim_ResNet18
layers=20
dataset=imagenet
root=/home/cs/bhuagroup/imagenet/
epochs=5
batch_size=32
s=0.001
prune_mode=resnet_slim_prune.py
prune=1
baseline_lr=0.1
sp_lr=0.01

echo "device=${device}"
echo "prefix=${prefix}"
echo "layers=${layers}"
echo "epochs=${epochs}"
echo "dataset=${dataset}"
echo "root=${root}"
echo "batch_size=${batch_size}"
echo "s=${s}"
echo "prune=${prune}"
echo "baseline_lr=${baseline_lr}"
echo "sp_lr=${sp_lr}"

############################################################
mkdir -p ${prefix}/baseline

echo "################ start baseline"
CUDA_VISIBLE_DEVICES=${device} python train.py \
    --layers ${layers} \
    -ckpt-dir ${prefix}/baseline/checkpoint \
    -log ${prefix}/baseline/runs \
    --epochs ${epochs} \
    --batch-size ${batch_size} \
    --lr ${baseline_lr} \
    --dataset ${dataset} \
    --root ${root}

echo "################ end baseline"


############################################################
mkdir -p ${prefix}/sp

echo "################ start sp"
CUDA_VISIBLE_DEVICES=${device} python train.py \
    -sr \
    --s ${s} \
    --layers ${layers} \
    -ckpt-dir ${prefix}/sp/checkpoint \
    -ckpt ${prefix}/baseline/checkpoint/last.pth \
    -log ${prefix}/sp/runs \
    --prune ${prune} \
    --epochs ${epochs} \
    --batch-size ${batch_size} \
    --lr ${sp_lr} \
    --dataset ${dataset} \
    --root ${root}
echo "################ end sp"

for percent in 0.5
do
    ############################################################
    mkdir -p ${prefix}/percent-${percent}

    echo "################ start prune"
    CUDA_VISIBLE_DEVICES=${device} python ${prune_mode} \
        --layers ${layers} \
        -ckpt ${prefix}/sp/checkpoint/last.pth \
        --saved-dir ${prefix}/percent-${percent}/pruned_model \
        --dataset ${dataset} \
        --percent ${percent}
    echo "################ end prune"

    # ###########################################################
    # mkdir -p ${prefix}/percent-${percent}/finetune_lr-0.1

    # CUDA_VISIBLE_DEVICES=${device} python train.py \
    #     --layers ${layers} \
    #     -ckpt-dir ${prefix}/percent-${percent}/finetune_lr-0.1/checkpoint \
    #     --cfg ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${layers}.cfg \
    #     -ckpt ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${layers}.pth \
    #     -log ${prefix}/percent-${percent}/finetune_lr-0.1/runs \
    #     --lr 0.1 \
    #     --epochs ${epochs} \
    #     --batch-size ${batch_size} \
    #     --dataset ${dataset} \
    #     --root ${root}

    ############################################################
    mkdir -p ${prefix}/percent-${percent}/finetune_lr-0.01

    CUDA_VISIBLE_DEVICES=${device} python train.py \
        --layers ${layers} \
        -ckpt-dir ${prefix}/percent-${percent}/finetune_lr-0.01/checkpoint \
        --cfg ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${layers}.cfg \
        -ckpt ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${layers}.pth \
        -log ${prefix}/percent-${percent}/finetune_lr-0.01/runs \
        --lr 0.01 \
        --epochs ${epochs} \
        --batch-size ${batch_size} \
        --dataset ${dataset} \
        --root ${root}

    # ###########################################################
    # mkdir -p ${prefix}/percent-${percent}/from_scratch

    # CUDA_VISIBLE_DEVICES=${device} python train.py \
    #     --layers ${layers} \
    #     -ckpt-dir ${prefix}/percent-${percent}/from_scratch/checkpoint \
    #     --cfg ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${layers}.cfg \
    #     -ckpt ${prefix}/percent-${percent}/pruned_model/pruned_percent-${percent}_${layers}.pth \
    #     -log ${prefix}/percent-${percent}/from_scratch/runs \
    #     --from-scratch \
    #     --epochs ${epochs} \
    #     --batch-size ${batch_size} \
    #     --dataset ${dataset} \
    #     --root ${root}
done

echo "################ finish!!!"