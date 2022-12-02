#!/usr/bin/env bash

EPOCHS=60
L2_LAMBDA=0.0
k=0.6

### MNIST

echo "Running MNIST pruning single layer"
for sigma in 1 2 3 4 5
do
    python  restore_compress.py --dataset mnist --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $sigma --pruning --single_layer
    sleep 2
done

echo "Running MNIST non-pruning single layer"
for sigma in 1 2 3 4 5
do
    python  restore_compress.py --dataset mnist --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $sigma --single_layer
    sleep 2
done

echo "Running MNIST pruning double layer"
for sigma in 1 2 3 4 5
do
    python  restore_compress.py --dataset mnist --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $sigma --pruning
    sleep 2
done

echo "Running MNIST non-pruning double layer"
for sigma in 1 2 3 4 5
do
    python  restore_compress.py --dataset mnist --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $sigma
    sleep 2
done

#### NEW

echo "Running CIFAR pruning single layer"
for sigma in 1 2 3 4 5
do
    python  restore_compress.py --dataset cifar10 --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $sigma --pruning --single_layer
    sleep 2
done

echo "Running CIFAR non-pruning single layer"
for sigma in 1 2 3 4 5
do
    python  restore_compress.py --dataset cifar10 --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $sigma --single_layer
    sleep 2
done

echo "Running CIFAR pruning double layer"
for sigma in 1 2 3 4 5
do
    python  restore_compress.py --dataset cifar10 --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $sigma --pruning
    sleep 2
done

echo "Running CIFAR non-pruning double layer"
for sigma in 1 2 3 4 5
do
    python  restore_compress.py --dataset cifar10 --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $sigma
    sleep 2
done