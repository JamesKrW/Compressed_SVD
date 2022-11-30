import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Union

from mlp import MLP


def hash_model(args: Any):
    """Hash the model's args to a unique name"""
    md5 = hashlib.md5(usedforsecurity=False)
    md5.update(json.dumps(vars(args), indent=2).encode("utf-8"))
    # md5.update(vars(args))
    dig = md5.hexdigest()

    print(json.dumps(vars(args)), dig)
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
    model.save(base_path / f"model-{dataset}-{dig}.pkl")


def load_model(
    model: MLP,
    args: Any,
    dataset: Union[Literal["mnist"], Literal["cifar10"]],
    base_path: Union[str, Path] = "./saved",
):
    """Load the model from a file with a unique name based on the model's"""
    base_path = Path(base_path)
    dig = hash_model(args)
    model.load(base_path / f"model-{dataset}-{dig}.pkl")


__all__ = ["hash_model", "save_model", "load_model"]
