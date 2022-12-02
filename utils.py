import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Union

from mlp import MLP


def get_hyperparams(args: Any):
    """Get the hyperparameters from the args"""
    arg_dict = OrderedDict(sorted(vars(args).copy().items()))
    arg_dict.pop("k", None)
    arg_dict.pop("dataset", None)
    arg_dict.pop("sigma", None)
    arg_dict.pop("pruning", None)
    arg_dict.pop("double_layer", None)
    arg_dict.pop("single_layer", None)
    return arg_dict


def print_hyperparams(args: Any):
    """Print the hyperparameters from the args"""
    arg_dict = get_hyperparams(args)
    print("Hyperparameters:")
    for key, value in arg_dict.items():
        print(f"  {key}: {value}")


def hash_model(args: Any):
    """Hash the model's args to a unique name"""
    md5 = hashlib.md5(usedforsecurity=False)
    arg_dict = get_hyperparams(args)
    md5.update(json.dumps(arg_dict).encode("utf-8"))
    dig = md5.hexdigest()
    return dig


def save_model(
    model: MLP,
    args: Any,
    dataset: Union[Literal["mnist"], Literal["cifar10"]],
    base_path: Union[str, Path] = "./saved",
):
    """Save the model to a file with a unique name based on the model's"""
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)
    dig = hash_model(args)
    print(f"Saving model to {base_path / f'model-{dataset}-{dig}.pkl'}")
    model.save_original(base_path / f"model-{dataset}-{dig}.pkl")


def load_model(
    model: MLP,
    args: Any,
    dataset: Union[Literal["mnist"], Literal["cifar10"]],
    base_path: Union[str, Path] = "./saved",
):
    """Load the model from a file with a unique name based on the model's"""
    base_path = Path(base_path)
    dig = hash_model(args)
    print(f"Loading model from {base_path / f'model-{dataset}-{dig}.pkl'}")
    model.load(base_path / f"model-{dataset}-{dig}.pkl")


__all__ = [
    "hash_model",
    "save_model",
    "load_model",
    "get_hyperparams",
    "print_hyperparams",
]
