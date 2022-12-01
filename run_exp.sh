#!/usr/bin/env bash

EPOCHS=60
L2_LAMBDA=0.0

echo "Running CIFAR pruning single layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python train_cifar10.py --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --pruning --single_layer
done

echo "Running CIFAR pruning double layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python train_cifar10.py --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --pruning
done

echo "Running MNIST pruning single layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python train_mnist.py --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --pruning --single_layer
done

echo "Running MNIST pruning double layer"
for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python train_mnist.py --k $k --epochs $EPOCHS --l2_lambda $L2_LAMBDA --pruning
done