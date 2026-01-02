#!/bin/bash
# Bash script to run all 4 experiments sequentially on CIFAR-100

echo "=========================================="
echo "  CIFAR-100 Attention Mechanism Experiments"
echo "=========================================="
echo ""

mkdir -p checkpoints logs results data
export CUDA_VISIBLE_DEVICES=0

# Experiment 1: Weak Baseline
echo "📊 Experiment 1: Weak Baseline (ResNet18, no aug, 50 epochs)"
python train.py --model resnet18 --epochs 50 --batch-size 128 --lr 0.1 --checkpoint-dir checkpoints/exp1 2>&1 | tee logs/exp1_train.log

# Experiment 2: Strong Baseline  
echo "📊 Experiment 2: Strong Baseline (ResNet18, with aug, 100 epochs)"
python train.py --model resnet18 --epochs 100 --batch-size 128 --lr 0.1 --checkpoint-dir checkpoints/exp2 2>&1 | tee logs/exp2_train.log

# Experiment 3: ResNet18 + SE
echo "📊 Experiment 3: ResNet18 + SE"
python train.py --model resnet18 --attention se --epochs 100 --batch-size 128 --lr 0.1 --checkpoint-dir checkpoints/exp3 2>&1 | tee logs/exp3_train.log

# Experiment 4: ResNet18 + CBAM (Proposed)
echo "📊 Experiment 4: ResNet18 + CBAM (Proposed Method)"
python train.py --model resnet18 --attention cbam --epochs 120 --batch-size 128 --lr 0.1 --checkpoint-dir checkpoints/exp4 2>&1 | tee logs/exp4_train.log

echo "✅ All Experiments Completed!"
echo "Results saved in checkpoints/ and logs/"
