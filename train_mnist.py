import argparse
from mlp import MLP
import numpy as np
from data_set import mnist
import metrics


@metrics.timing
def main(args):
    epochs = args.epochs
    batch_size = args.batch_size
    train_set, valid_set, test_set = mnist(args.data_path, one_hot=True)
    X_train = train_set[0]
    Y_train = train_set[1]

    model = MLP(args.model_shape, lr=args.learning_rate, l2_lambda=args.l2_lambda)

    with metrics.Benchmark("Training"):
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
                # print('Epoch[{}], Step[{}], loss = {}'.format(epoch, i, loss))

                if (i + 1) % 100 == 0:
                    accuracy = model.validate(valid_set[0], valid_set[1])
                    print(
                        "Epoch[{}/{}] \t Step[{}/{}] \t Loss = {:.6f} \t Acc = {:.3f}".format(
                            epoch + 1, epochs, i + 1, steps, loss, accuracy
                        )
                    )
            model.lr_step()

    with metrics.Benchmark("Validation Before Compression"):
        test_accuracy = model.validate(test_set[0], test_set[1])
    print("==== BEFORE COMPRESSION ====")
    print("Test Accuracy = ", test_accuracy)
    print(f"Memory size before compression: {metrics.get_mem_size_kb(model)} KB")
    print(
        f"Persisted size before compression: {metrics.get_persisted_size_kb(model)} KB"
    )
    print()

    model.compress_mlp(k=5, double_layer=True)
    with metrics.Benchmark("Validation After Compression"):
        test_accuracy = model.validate(test_set[0], test_set[1])
    print("==== AFTER COMPRESSION ====")
    print("Test Accuracy", test_accuracy)
    print(f"Memory size after compression: {metrics.get_mem_size_kb(model)} KB")
    print(
        f"Persisted size after compression: {metrics.get_persisted_size_kb(model)} KB"
    )
    print()

    metrics.Benchmark.print_all_benchmarks()


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=60, type=int)
parser.add_argument("--data_path", default="./data/mnist.pkl.gz")
parser.add_argument("--model_shape", default=[784, 20, 20, 20, 10], type=list)
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--l2_lambda", default=0.00, type=float)
args = parser.parse_args()

# trainable arguments:shape=[784, 300, 150,75,10],lr=0.01
# trainable arguments:shape=[784, 20, 20,20,10],lr=0.01
main(args)
