import argparse
from datetime import datetime

import metrics
import numpy as np
from data_set import cifar10
from mlp import MLP


@metrics.timing
def main(args):
    epochs = args.epochs
    batch_size = args.batch_size
    train_set, valid_set, test_set = cifar10(
        args.data_path, one_hot=True
    )
    X_train = train_set[0]
    Y_train = train_set[1]

    model = MLP(
        args.model_shape,
        batch_size=batch_size,
        lr=args.learning_rate,
        l2_lambda=args.l2_lambda,
    )

    metric = metrics.Metric()

    metric.add("time", datetime.now())
    metric.add("dataset", "cifar")
    metric.add("k", args.k)
    metric.add("shape", args.model_shape)
    metric.add("batch_size", args.batch_size)
    metric.add("lr", args.learning_rate)
    metric.add("epochs", args.epochs)
    metric.add("l2_lambda", args.l2_lambda)

    with metrics.Timer("Training") as t:
        for epoch in range(epochs):
            indexes = np.arange(len(X_train))
            steps = len(X_train) // batch_size
            np.random.shuffle(indexes)
            for i in range(steps):
                ind = indexes[i * batch_size : (i + 1) * batch_size]
                x = X_train[ind]
                y = Y_train[ind]
                loss = model(x, y)
                model.backward()

                if (i + 1) % 100 == 0:
                    accuracy = model.validate(
                        valid_set[0], valid_set[1]
                    )
                    print(
                        "Epoch[{}/{}] \t Step[{}/{}] \t Loss = {:.6f} \t Acc = {:.3f}".format(
                            epoch + 1,
                            epochs,
                            i + 1,
                            steps,
                            loss,
                            accuracy,
                        )
                    )
            model.lr_step()

    metric.add("train_time", t.get(), "s")

    metrics.run_metrics(
        metric, model, test_set, after_compression=False
    )
    model.compress_mlp(k=args.k, double_layer=True)
    metrics.run_metrics(
        metric, model, test_set, after_compression=True
    )

    metric.show()
    metric.save_to_csv("cifar.csv")


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--k", default=5, type=int)
parser.add_argument(
    "--data_path", default="./data/cifar-10-binary.tar.gz"
)
parser.add_argument(
    "--model_shape", default=[3072, 20, 20, 10], type=list
)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--l2_lambda", default=0.01, type=float)
args = parser.parse_args()


main(args)