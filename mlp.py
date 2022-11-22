import numpy as np
import pickle


class MLP:
    def __init__(
        self,
        size=(784, 100, 10),
        lr: float = 0.01,
        lr_ratio: float = 0.9,
        l2_lambda: float = 0.0,
    ):
        assert isinstance(size, list) or isinstance(
            size, tuple
        ), "size must be a list or tuple."
        assert len(size) > 1, "the length of size must be greater than 1."

        self.lr = lr
        self.lr_ratio = lr_ratio
        self.weights = []
        self.bias = []
        self.l2_lambda = l2_lambda
        self.compressed = False
        self.double_layer = False

        for i in range(1, len(size)):
            self.weights.append(np.random.normal(0, 0.01, (size[i - 1], size[i])))
            self.bias.append(np.random.normal(0, 0.01, (1, size[i])))

    def ReLU(self, z):
        return np.maximum(z, 0.0)

    def softmax(self, z):
        z_temp = np.exp(z)
        return z_temp / np.sum(z_temp, axis=1, keepdims=True)

    def recover(self, weights, bias):
        self.weights = weights
        self.bias = bias
        return self

    def _loss(self, Y):
        cross_entropy_loss = -np.sum(Y * np.log(self.a[-1])) / len(Y)

        sum_square = 0.0

        for w in self.weights:
            sum_square += np.sum(np.square(w))

        l2_loss = sum_square * self.l2_lambda / (2 * len(Y))

        return cross_entropy_loss + l2_loss

    def __call__(self, X, Y):
        self.target = Y

        self.z = []
        self.a = [X]
        for w, b in zip(self.weights, self.bias):
            self.z.append(self.a[-1] @ w + b)
            self.a.append(self.ReLU(self.z[-1]))
        self.a[-1] = self.softmax(self.z[-1])

        return self._loss(Y)

    def backward(self):
        W_grad = []
        b_grad = []
        delta = (self.a[-1] - self.target) / len(self.target)
        for i in range(len(self.weights) - 1, -1, -1):
            W_grad.insert(0, self.a[i].T @ delta)
            b_grad.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i == 0:
                break
            delta = (
                delta @ (self.weights[i].T) * (self.z[i - 1] > 0.0)
            )  # * self.a[i] * (1.0 - self.a[i])

        # update trained parameters
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * W_grad[i]
            self.bias[i] -= self.lr * b_grad[i]

        return self

    def predict(self, X):
        a = X
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.bias[i]
            if self.double_layer:
                if np.sum(self.bias[i] ** 2) < 1e-6:
                    a = z
                    continue
            a = self.ReLU(z)

        return self.softmax(z)

    def validate(self, X, Y):
        pred = np.argmax(self.predict(X), axis=1)
        org = np.argmax(Y, axis=1)
        acc = sum(pred == org) / len(pred)
        return acc

    def lr_step(self):
        self.lr *= self.lr_ratio
        return self

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump((self.weights, self.bias), f)
            f.close()
        return self

    def load(self, path: str):
        with open(path, "rb") as f:
            self.weights, self.bias = pickle.load(f)
            f.close()
        return self

    def truncated_svd(self, W, k: int):
        U, sigma, V = np.linalg.svd(W, full_matrices=True, compute_uv=True)
        Sigma = np.zeros_like(W)

        for i in range(k):
            if len(sigma) > k:
                Sigma[i][i] = sigma[i]

        W = U @ Sigma @ V
        U = U @ Sigma
        return W, U, V

    def compress_mlp(self, k: int, double_layer: bool = False):
        weights = []
        bias = []

        if self.compressed:
            return self

        for i, w in enumerate(self.weights):
            W, U, V = self.truncated_svd(w, k)
            if double_layer:
                print("LAYER {}:".format(i))
                weights.append(U)
                print(U.shape)
                weights.append(V)
                print(V.shape)
                bias.append(np.zeros((1, U.shape[1])))
                print(np.zeros((1, U.shape[1])).shape)
                bias.append(self.bias[i])
                print(self.bias[i].shape)
                print()
            else:
                weights.append(W)
                bias.append(self.bias[i])

        self.double_layer = double_layer
        self.compressed = True

        return self.recover(weights, bias)
