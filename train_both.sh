#!/usr/bin/env bash

EPOCHS=60
L2_LAMBDA=0.0

### MNIST

echo "Training MNIST "
python train_mnist.py --epochs $EPOCHS --l2_lambda $L2_LAMBDA 
sleep 5


### CIFAR10

echo "Training CIFAR10"
python train_cifar10.py --epochs $EPOCHS --l2_lambda $L2_LAMBDA