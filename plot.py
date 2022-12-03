import argparse
from itertools import cycle
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

COLUMNS = ["test_acc", "test_time", "mem_size", "persisted_size"]
GROUPED_COLUMNS = [
    ["test_acc", "test_time"],
    "mem_size",
    "persisted_size",
]
COLUMN_HUMANIZE = [
    "Accuracy",
    "Time (in ms)",
    "Memory Size (in KB)",
    "Persistence Size (in KB)",
]
FIG_SIZE = (4, 3)
FIXED_LISTS = ["fixed_k", "fixed_sigma"]

# each plot is one line
MNIST_FILES = [
    "mnist_non_pruned_double_layer.csv",
    "mnist_pruned_double_layer.csv",
    "mnist_pruned_single_layer.csv",
    "mnist_non_pruned_single_layer.csv",
]
CIFAR10_FILES = [
    "cifar10_non_pruned_double_layer.csv",
    "cifar10_pruned_double_layer.csv",
    "cifar10_pruned_single_layer.csv",
    "cifar10_non_pruned_single_layer.csv",
]

ORIGINAL_COLOR = "green"
ORIGINAL_MARKER = "o"
ALL_COLORS = ["maroon", "orange", "blue", "black"]
ALL_MARKERS = ["v", "s", "D", "X"]

# COLOR = cycle(ALL_COLORS[: len(MNIST_FILES) + 1])
# MARKER = cycle(ALL_MARKERS[: len(ALL_MARKERS) + 1])


@ticker.FuncFormatter
def second_to_ms_formatter(x, pos):
    return f"{x * 1000:g} ms"


def plot_column_from_dfs(
    ax: plt.Axes,
    dfs: List[pd.DataFrame],
    original: pd.DataFrame,
    y: str,
    x: str = "k",
    fixed: str = "k",
):
    if y in ["mem_size", "persisted_size"] and "k" not in fixed:
        ax.axhline(
            original[y][0],
            c=ORIGINAL_COLOR,
            linestyle="--",
            label="Original",
            markersize=3,
            marker=ORIGINAL_MARKER,
        )

    for i, df in enumerate(dfs):
        if y in ["test_acc", "test_time"] and i == 0:
            avg = df[f"bc_{y}"].max()
            ax.axhline(
                avg,
                # x=x,
                # ax=ax,
                c=ORIGINAL_COLOR,
                linestyle="--",
                label="Original",
                markersize=3,
                marker=ORIGINAL_MARKER,
            )
            # df.plot(
            #     x=x,
            #     y=f"bc_{y}",
            #     ax=ax,
            #     c=ORIGINAL_COLOR,
            #     linestyle="--",
            #     label="Original",
            #     markersize=3,
            #     marker=ORIGINAL_MARKER,
            # )

        name = ""
        if (df["pruning"] == 0).all():
            name += "Unpruned"
        elif (df["pruning"] == 1).all():
            name += "Pruned"
        else:
            raise ValueError("Invalid pruning value")

        if (df["double_layer"] == 0).all():
            name += " Single Layer"
        elif (df["double_layer"] == 1).all():
            name += " Double Layer"
        else:
            raise ValueError("Invalid layer value")

        df.plot(
            x=x,
            y=f"ac_{y}",
            ax=ax,
            c=ALL_COLORS[i],
            marker=ALL_MARKERS[i],
            markersize=3,
            label=name,
        )
        # original.plot(
        #     x=x,
        #     y=f"{y}",
        #     ax=ax,
        #     c=ORIGINAL_COLOR,
        #     linestyle="--",
        #     label="Original",
        #     markersize=3,
        #     marker=ORIGINAL_MARKER,
        # )


def plot_single_dataset(
    dataset: str,
    data_path: str,
    base_output_path: str,
    format: str = "eps",
    combine: bool = False,
):
    assert dataset in ["mnist", "cifar10"], "Dataset must be either mnist or cifar10"

    data_path = Path(data_path)

    original_dataset_pd = pd.read_csv(data_path / f"{dataset}_original.csv")

    for fixed in FIXED_LISTS:
        files = MNIST_FILES if dataset == "mnist" else CIFAR10_FILES

        output_path = Path(base_output_path) / fixed
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"Output path: {output_path}")

        dfs = []

        x = fixed.removeprefix("fixed_")
        x = "k" if x == "sigma" else "sigma"

        if "sigma" in fixed:
            dfs = [pd.read_csv(data_path / fixed / f) for f in files]
        else:
            dfs = [pd.read_csv(data_path / fixed / f) for f in files if "non" not in f]

        print(f"X: {x}")

        # n = 1
        if combine:
            fig, axs = plt.subplots(
                1,
                2,
                figsize=(FIG_SIZE[0] * 2, FIG_SIZE[1]),
                constrained_layout=True,
            )

            for i, col in enumerate(COLUMNS[:2]):
                axs[i].set_title(COLUMN_HUMANIZE[i])
                plot_column_from_dfs(
                    axs[i],
                    dfs,
                    original=original_dataset_pd,
                    y=col,
                    x=x,
                )
                axs[i].legend().set_visible(False)
                axs[i].set_xlabel("N" if x == "sigma" else "k")

            handles, labels = axs[-1].get_legend_handles_labels()
            # order = [2, 0, 1, 3, 4]
            # order = [2, 0, 1, 3]
            # order = [2, 0, 1]
            # print(handles, labels)
            lgd = fig.legend(
                handles,
                labels,
                # [handles[idx] for idx in order],
                # [labels[idx] for idx in order],
                bbox_to_anchor=(0.5, -0.2),
                loc="lower center",
                ncol=3,
            )

            fig.savefig(
                output_path / f"{dataset}_{fixed}_test_acc_time.{format}",
                format=format,
                dpi=1200,
                bbox_extra_artists=(lgd,),
                bbox_inches="tight",
            )
            plt.close(fig)

            # ------------------------------

            fig, axs = plt.subplots(
                1,
                2,
                figsize=(FIG_SIZE[0] * 2, FIG_SIZE[1]),
                constrained_layout=True,
            )

            for i, col in enumerate(COLUMNS[2:]):
                axs[i].set_title(COLUMN_HUMANIZE[i + 2])
                plot_column_from_dfs(
                    axs[i], dfs, original=original_dataset_pd, y=col, x=x, fixed=fixed
                )
                axs[i].legend().set_visible(False)
                axs[i].set_xlabel("N" if x == "sigma" else "k")

            handles, labels = axs[-1].get_legend_handles_labels()
            # order = [2, 0, 1, 3, 4]
            # order = [2, 0, 1, 3]
            # order = [2, 0, 1]
            # print(handles, labels)
            lgd = fig.legend(
                handles,
                labels,
                # [handles[idx] for idx in order],
                # [labels[idx] for idx in order],
                bbox_to_anchor=(0.5, -0.2),
                loc="lower center",
                ncol=3,
            )

            fig.savefig(
                output_path / f"{dataset}_{fixed}_size.{format}",
                format=format,
                dpi=1200,
                bbox_extra_artists=(lgd,),
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            for col in COLUMNS:
                out_path = output_path / f"{fixed}_{dataset}_{col}.{format}"
                out_path.parent.mkdir(exist_ok=True, parents=True)
                fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, constrained_layout=True)
                plot_column_from_dfs(
                    ax, dfs, original=original_dataset_pd, y=col, x=x, fixed=fixed
                )
                fig.savefig(
                    out_path,
                    format=format,
                    dpi=1200,
                )
                plt.close(fig)


def main(
    dataset: str,
    data_path: str,
    output_path: str,
    format: str = "eps",
    combine: bool = False,
):
    assert dataset in [
        "mnist",
        "cifar10",
        "all",
    ], "Dataset must be either mnist or cifar10 or all"

    if dataset == "all":
        plot_single_dataset(
            "mnist", data_path, output_path, format=format, combine=combine
        )
        plot_single_dataset(
            "cifar10", data_path, output_path, format=format, combine=combine
        )
    else:
        plot_single_dataset(
            dataset, data_path, output_path, format=format, combine=combine
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", default="./plot_data", type=str)
    parser.add_argument("-o", "--output", default="./plots", type=str)
    parser.add_argument("--format", default="eps", type=str)
    parser.add_argument("-c", "--combine", action="store_true")
    parser.add_argument("-d", "--dataset", type=str, required=True)
    args = parser.parse_args()
    main(
        args.dataset, args.folder, args.output, format=args.format, combine=args.combine
    )
