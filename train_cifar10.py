import argparse
from mlp import MLP
import numpy as np
from data_set import cifar10
import metrics


@metrics.timing
def train_cifar10(args):
    epochs = args.epochs
    batch_size = args.batch_size
    train_set, valid_set, test_set = cifar10(args.data_path, one_hot=True)
    X_train = train_set[0]
    Y_train = train_set[1]

    model = MLP(args.model_shape, lr=args.learning_rate)

    for epoch in range(epochs):
        indexs = np.arange(len(X_train))
        steps = len(X_train) // batch_size
        np.random.shuffle(indexs)
        for i in range(steps):
            ind = indexs[i * batch_size : (i + 1) * batch_size]
            x = X_train[ind]
            y = Y_train[ind]
            loss = model(x, y)
            model.backward()
            # print('Epoch[{}], Step[{}], loss = {}'.format(epoch, i, loss))

            if (i + 1) % 100 == 0:
                accuracy = model.validate(valid_set[0], valid_set[1])
                print(
                    "Epoch[{}/{}] \t Step[{}/{}] \t Loss = {:.6f} \t Acc = {:.3f}".format(
                        epoch + 1, epochs, i + 1, steps, loss, accuracy
                    )
                )
        model.lr_step()
    print("Test Accuracy = ", model.validate(test_set[0], test_set[1]))


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--data_path", default="./data/cifar-10-binary.tar.gz")
parser.add_argument("--model_shape", default=[3072, 20, 20, 10], type=list)
parser.add_argument("--learning_rate", default=0.01, type=float)
args = parser.parse_args()


train_cifar10(args)
