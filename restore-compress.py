import argparse
from datetime import datetime

import metrics
from dataset import cifar10, mnist
from mlp import MLP
from utils import load_model

# THIS IS ONLY FINE IF YOU WANT TO GET PERSISTED SIZE
# MEMORY SIZE WON'T WORK FINE BECAUSE RESTORED ORIGINAL MODEL
# BEFORE COMPRESSION WON'T DO ANY CALCULATION THUS NO MEMORY IS ALLOCATED
# COMPRESSED MODEL ON THE OTHER HAND WILL DO CALCULATION AND ALLOCATE MEMORY
# THUS CREATING MORE MEMORY USAGE THAN ORIGINAL MODEL


def main(args):
    dataset = args.dataset
    del args.dataset
    batch_size = args.batch_size
    if dataset == "mnist":
        _, _, test_set = mnist(args.data_path, one_hot=True)
    elif dataset == "cifar":
        _, _, test_set = cifar10(args.data_path, one_hot=True)

    metric = metrics.Metric()

    metric.add("time", datetime.now())
    metric.add("dataset", "cifar")
    metric.add("k", args.k)
    metric.add("shape", args.model_shape)
    metric.add("batch_size", args.batch_size)
    metric.add("lr", args.learning_rate)
    metric.add("epochs", args.epochs)
    metric.add("l2_lambda", args.l2_lambda)

    model = MLP(
        args.model_shape,
        batch_size=batch_size,
        lr=args.learning_rate,
        l2_lambda=args.l2_lambda,
    )

    metric.add("train_time", 0, "s")

    model = load_model(
        model, args, dataset=dataset, base_path="./saved"
    )

    metrics.run_metrics(
        metric, model, test_set, after_compression=False
    )
    model.compress_mlp(k=args.k, double_layer=True)
    metrics.run_metrics(
        metric, model, test_set, after_compression=True
    )

    metric.show()
    metric.save_to_csv(f"{dataset}.csv")


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=60, type=int)
parser.add_argument("--dataset", default="mnist", type=str)
parser.add_argument("--k", default=5, type=int)
parser.add_argument("--data_path", default="./data/mnist.pkl.gz")
parser.add_argument(
    "--model_shape", default=[784, 20, 20, 10], type=list
)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--l2_lambda", default=0.0, type=float)
args = parser.parse_args()

main(args)
