#!/usr/bin/env bash

EPOCHS=60
L2_LAMBDA=0.0
SIGMA=2.0

### MNIST

echo "Running MNIST pruning single layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python restore_compress.py --dataset mnist --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $SIGMA --pruning --single_layer
    sleep 2
done

echo "Running MNIST non-pruning single layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python restore_compress.py --dataset mnist --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $SIGMA --single_layer
    sleep 2
done

echo "Running MNIST pruning double layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python restore_compress.py --dataset mnist --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $SIGMA --pruning
    sleep 2
done

echo "Running MNIST non-pruning double layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python restore_compress.py --dataset mnist --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $SIGMA
    sleep 2
done

#### CIFAR

echo "Running CIFAR pruning single layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python restore_compress.py --dataset cifar10 --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $SIGMA --pruning --single_layer
    sleep 2
done

echo "Running CIFAR non-pruning single layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python restore_compress.py --dataset cifar10 --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $SIGMA --single_layer
    sleep 2
done

echo "Running CIFAR pruning double layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python restore_compress.py --dataset cifar10 --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $SIGMA --pruning
    sleep 2
done

echo "Running CIFAR non-pruning double layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python restore_compress.py --dataset cifar10 --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --sigma $SIGMA
    sleep 2
done