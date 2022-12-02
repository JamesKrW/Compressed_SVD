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
    assert args.dataset in [
        "mnist",
        "cifar10",
    ], "Dataset must be either mnist or cifar10"
    dataset = args.dataset

    batch_size = args.batch_size
    if dataset == "mnist":
        args.data_path = "./data/mnist.pkl.gz"
        _, _, test_set = mnist(args.data_path, one_hot=True)
        args.model_shape = [784, 20, 20, 10]
    else:
        args.data_path = "./data/cifar-10-binary.tar.gz"
        _, _, test_set = cifar10(args.data_path, one_hot=True)
        args.model_shape = [3072, 20, 20, 10]

    metric = metrics.Metric()

    metric.add("time", datetime.now())
    metric.add("dataset", dataset)
    metric.add("k", args.k)
    metric.add("shape", args.model_shape)
    metric.add("batch_size", args.batch_size)
    metric.add("lr", args.learning_rate)
    metric.add("epochs", args.epochs)
    metric.add("l2_lambda", args.l2_lambda)
    metric.add("double_layer", int(not args.single_layer))
    metric.add("pruning", int(args.pruning))
    metric.add("sigma", args.sigma)

    model = MLP(
        args.model_shape,
        batch_size=batch_size,
        lr=args.learning_rate,
        l2_lambda=args.l2_lambda,
    )

    metric.add("train_time", 0, "s")

    load_model(model, args, dataset=dataset, base_path="./saved")

    metrics.run_metrics(metric, model, test_set, after_compression=False)

    if args.pruning:
        model.sigma_pruning(lambda_=args.sigma)
    model.compress_mlp(k=args.k, double_layer=True)
    metrics.run_metrics(metric, model, test_set, after_compression=True)

    metric.show()
    pruning_fl = "pruned" if args.pruning else "non_pruned"
    layer_fl = "single_layer" if args.single_layer else "double_layer"
    filename = f"{dataset}_{pruning_fl}_{layer_fl}"
    metric.save_to_csv(f"{filename}.csv")


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=60, type=int)
parser.add_argument("--k", default=0.5, type=float)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--l2_lambda", default=0.00, type=float)
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

main(args)
