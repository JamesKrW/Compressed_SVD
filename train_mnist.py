import argparse
from datetime import datetime
from pathlib import Path

import metrics
import numpy as np
from dataset import mnist
from mlp import MLP
from utils import hash_model, load_model, save_model

DATASET = "mnist"


@metrics.timing
def main(args):
    epochs = args.epochs
    batch_size = args.batch_size
    train_set, valid_set, test_set = mnist(args.data_path, one_hot=True)
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
    metric.add("dataset", DATASET)
    metric.add("k", args.k)
    metric.add("shape", args.model_shape)
    metric.add("batch_size", args.batch_size)
    metric.add("lr", args.learning_rate)
    metric.add("epochs", args.epochs)
    metric.add("l2_lambda", args.l2_lambda)
    metric.add("double_layer", int(not args.single_layer))
    metric.add("pruning", int(args.pruning))
    metric.add("sigma", args.sigma)

    if not Path(f"./saved/model-{DATASET}-{hash_model(args)}.pkl").exists():
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
                        accuracy = model.validate(valid_set[0], valid_set[1])
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
        save_model(model=model, args=args, dataset=DATASET, base_path="./saved")
        metrics.run_metrics(metric, model, test_set, after_compression=False)
    else:
        metric.add("train_time", 0, "s")
        load_model(model, args, dataset=DATASET, base_path="./saved")

    if args.pruning:
        model.sigma_pruning(lambda_=args.sigma)
    model.compress_mlp(k=args.k, double_layer=not args.single_layer)
    metrics.run_metrics(metric, model, test_set, after_compression=True)

    metric.show()
    pruning_fl = "pruned" if args.pruning else "non_pruned"
    layer_fl = "single_layer" if args.single_layer else "double_layer"
    filename = f"mnist_{pruning_fl}_{layer_fl}"
    metric.save_to_csv(f"{filename}.csv")


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=60, type=int)
parser.add_argument("--data_path", default="./data/mnist.pkl.gz")
parser.add_argument("--model_shape", default=[784, 20, 20, 10], type=list)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--l2_lambda", default=0.0, type=float)
parser.add_argument("--k", default=0.5, type=float)
parser.add_argument("--sigma", default=2.0, type=float)
parser.add_argument(
    "--pruning",
    help="Enable pruning",
    action="store_true",
)
parser.add_argument(
    "--single_layer",
    help="Enable single layer",
    action="store_true",
)
args = parser.parse_args()

# trainable arguments:shape=[784, 300, 150, 75, 10], lr=0.01
# trainable arguments:shape=[784, 20, 20, 20, 10], lr=0.01
# trainable arguments:shape=[784, 20, 20, 10], lr=0.01, l2_lambda=0.01
main(args)
