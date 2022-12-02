import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Union

import numpy.typing as npt
import pandas as pd
from mlp import MLP


def get_mem_size(obj, seen=None):
    """Recursively finds size of objects (in bytes)"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_mem_size(v, seen) for v in obj.values()])
        size += sum([get_mem_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_mem_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_mem_size(i, seen) for i in obj])
    return size


def get_mem_size_kb(obj):
    """Recursively finds size of objects (in KB)"""
    return get_mem_size(obj) / 1024


def get_persisted_size(obj):
    """Get the size of the object (in bytes) after it has been persisted to disk."""
    id = uuid.uuid4()
    path = Path("/tmp") / str(id)
    path = path.with_suffix(".pkl")
    obj.save(path)
    size = path.stat().st_size
    path.unlink()
    return size


def get_persisted_size_kb(obj):
    """Get the size of the object (in KB) after it has been persisted to disk."""
    return get_persisted_size(obj) / 1024


def timing(func):
    """Calculate the execute time of the given func"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        st = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"'{func.__name__}()' starts at {st}")
        result = func(*args, **kwargs)
        end = time.perf_counter()
        ed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tot_time = end - start
        tot_time = float(f"{tot_time:.4f}")
        print(f"'{func.__name__}()' ends at {ed} and takes {tot_time} seconds.")
        func.tot_time = tot_time  # add new variable to func
        return result, tot_time

    return wrapper


class Timer:
    def __init__(self, msg: str, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        st = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print()
        print(f"'{self.msg}' starts at {st}")
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        t = time.perf_counter() - self.start
        ed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t = float(f"{t:.4f}")
        print(f"'{self.msg}' ends at {ed} and takes {t} seconds.")
        print()
        self.time = t

    def get(self):
        return self.time


@dataclass
class MetricValue:
    value: Any
    unit: str = ""

    def __str__(self):
        return f"{self.value}"


class Metric(object):
    """A context manager to measure the metrics"""

    def __init__(self, fmt="%0.3g"):
        self.fmt = fmt
        self._metrics: dict[str, MetricValue] = {}

    def add(self, key: str, v: float, unit: str = ""):
        self._metrics[key] = MetricValue(v, unit)

    def show(self, tabulate: bool = True):
        if tabulate:
            from tabulate import tabulate

            res = []
            for key, v in self._metrics.items():
                res.append([key, v.value, v.unit])
            print(
                tabulate(
                    res,
                    headers=["Message", "Value", "Unit"],
                    tablefmt="heavy_outline",
                )
            )
            return

        print("===== ALL METRICS =====")
        for key, v in self._benchmarks.items():
            print(f"{key}: {v.value} {v.unit}")

    # @classmethod
    def get_all(self):
        return self._metrics

    def get(self, key: str):
        return self._metrics[key]

    def save_to_csv(self, path: Union[str, Path]):
        headers = list(self._metrics.keys())
        df = pd.DataFrame([self._metrics])
        if not os.path.isfile(path):
            df.to_csv(path, header=headers, index=False)
        else:
            csv_file = open(path, "a")
            df.to_csv(
                csv_file,
                mode="a",
                index=False,
                header=csv_file.tell() == 0,
            )
            csv_file.close()


def run_metrics_original_model(
    metric: Metric,
    model: MLP,
    # test_set: npt.ArrayLike,
):
    """Run all the metrics and save them to the given metric object"""
    # with Timer("Test") as t:
    #     test_accuracy = model.validate(test_set[0], test_set[1])
    metric.add("original", 1)
    # metric.add("test_time", t.get(), "s")
    # print("Test Accuracy = ", test_accuracy)
    # metric.add("test_acc", test_accuracy)
    mem_size = get_mem_size_kb(model.metadata)
    print(f"Memory size: {mem_size} KB")
    metric.add("mem_size", mem_size)
    persisted_size = get_persisted_size_kb(model)
    print(f"Persisted size: {persisted_size} KB")
    metric.add("persisted_size", persisted_size)
    print()


def run_metrics(
    metric: Metric,
    model: MLP,
    test_set: npt.ArrayLike,
    after_compression: bool = False,
):
    """Run all the metrics and save them to the given metric object"""
    prefix = "bc"
    if after_compression:
        print("===== AFTER COMPRESSION =====")
        prefix = "ac"
    else:
        print("==== BEFORE COMPRESSION ====")

    with Timer("Test") as t:
        test_accuracy = model.validate(test_set[0], test_set[1])
    metric.add("original", int(not after_compression))
    metric.add(f"{prefix}_test_time", t.get(), "s")
    print("Test Accuracy = ", test_accuracy)
    metric.add(f"{prefix}_test_acc", test_accuracy)
    mem_size = get_mem_size_kb(model.metadata)
    print(f"Memory size: {mem_size} KB")
    metric.add(f"{prefix}_mem_size", mem_size)
    persisted_size = get_persisted_size_kb(model)
    print(f"Persisted size: {persisted_size} KB")
    metric.add(f"{prefix}_persisted_size", persisted_size)
    print()


__all__ = [
    "get_mem_size",
    "get_mem_size_kb",
    "get_persisted_size",
    "get_persisted_size_kb",
    "timing",
    "Metric",
]
