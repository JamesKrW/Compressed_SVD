import argparse
from itertools import cycle
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

COLUMNS = ["test_acc", "test_time", "mem_size", "persisted_size"]
COLUMN_HUMANIZE = [
    "Accuracy",
    "Time (in ms)",
    "Memory Size (in KB)",
    "Persisted Size (in KB)",
]
FIG_SIZE = (4, 3)
# FILES = ["mnist.csv", "cifar.csv"]

# each plot is one line
MNIST_FILES = [
    "mnist_non_pruned_double_layer.csv",
    "mnist_pruned_double_layer.csv",
    "mnist_pruned_single_layer.csv",
    "mnist_non_pruned_single_layer.csv",
]
CIFAR10_FILES = [
    "cifar_non_pruned_double_layer.csv",
    "cifar_pruned_double_layer.csv",
    "cifar_pruned_single_layer.csv",
    "cifar_non_pruned_single_layer.csv",
]

ORIGINAL_COLOR = "green"
ORIGINAL_MARKER = "o"
ALL_COLORS = ["maroon", "orange", "purple", "black"]
ALL_MARKERS = ["v", "s", "D", "X"]

# COLOR = cycle(ALL_COLORS[: len(MNIST_FILES) + 1])
# MARKER = cycle(ALL_MARKERS[: len(ALL_MARKERS) + 1])


@ticker.FuncFormatter
def second_to_ms_formatter(x, pos):
    return f"{x * 1000:g} ms"


def plot_column(df: pd.DataFrame, y: str, x: str = "k"):
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, constrained_layout=True)
    # print(df.loc[:, y.replace("ac_", "bc_")])
    # ax.axhline(
    #     y=df.loc[:, y.replace("ac_", "bc_")].values[0],
    #     color="r",
    #     linestyle="-",
    # )
    df.plot(
        x=x,
        y=f"bc_{y}",
        ax=ax,
        c="green",
        linestyle="--",
        label="Original",
    )
    df.plot(x=x, y=f"ac_{y}", ax=ax, c="maroon", label="Compressed")

    if "acc" in y:
        pass
        # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=5))
        # ax.set_ylabel("Accuracy")
    if "size" in y:
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g} KB"))
        # ax.set_ylabel("in KB")
    if "time" in y:
        ax.yaxis.set_major_formatter(second_to_ms_formatter)
        # ax.set_ylabel("in ms")
        # ax.set_ylabel("Size in (KB)")

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    # fig.suptitle(title)
    # fig.tight_layout()
    # ax.legend().set_visible(False)
    return fig


# def plot_all_columns(
#     df: pd.DataFrame,
#     ys: List[str],
#     x: str = "k",
#     with_title: bool = False,
# ):
#     n = len(ys)
#     fig, axs = plt.subplots(
#         1,
#         n,
#         figsize=(FIG_SIZE[0] * n, FIG_SIZE[1]),
#         constrained_layout=True,
#     )

#     for i, y in enumerate(ys):
#         df.plot(
#             x=x,
#             y=f"bc_{y}",
#             ax=axs[i],
#             c="green",
#             linestyle="--",
#             label="Original",
#         )
#         df.plot(
#             x=x,
#             y=f"ac_{y}",
#             ax=axs[i],
#             c="maroon",
#             label="Compressed",
#         )

#         if "acc" in y:
#             pass
#             # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=5))
#             # if with_title:
#             #     axs[i].set_ylabel("Accuracy")
#         if "size" in y:
#             axs[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g} KB"))
#             # ax.set_ylabel("in KB")
#         if "time" in y:
#             axs[i].yaxis.set_major_formatter(second_to_ms_formatter)
#             # ax.set_ylabel("in ms")
#             # ax.set_ylabel("Size in (KB)")

#         axs[i].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

#     # if with_title:
#     #     fig.suptitle("mnist")
#     # fig.suptitle(title)
#     # fig.tight_layout()
#     # ax.legend().set_visible(False)
#     return fig


# def plot(csv: Union[str, Path], output_path: str, format: str = "eps"):
#     dataset = "mnist" if "mnist" in csv else "cifar10"
#     # output_path = BASE_OUTPUT_PATH
#     output_path = Path(output_path)
#     output_path.mkdir(exist_ok=True, parents=True)
#     df = pd.read_csv(csv)

#     for col in COLUMNS:
#         fig = plot_column(df, col, x="k")
#         fig.savefig(
#             output_path / f"{dataset}_{col}.{format}",
#             format=format,
#             dpi=1200,
#         )
#         plt.close(fig)

#     fig = plot_all_columns(df, COLUMNS, x="k", with_title="mnist" in csv)
#     fig.savefig(
#         output_path / f"{dataset}_all.{format}",
#         format=format,
#         dpi=1200,
#     )


def plot_column_from_dfs(ax: plt.Axes, dfs: List[pd.DataFrame], y: str, x: str = "k"):
    for i, df in enumerate(dfs):
        if i == len(dfs) - 2:
            df.plot(
                x=x,
                y=f"bc_{y}",
                ax=ax,
                c=ORIGINAL_COLOR,
                linestyle="--",
                label="Original",
                markersize=3,
                marker=ORIGINAL_MARKER,
            )

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


def plot_single_dataset(
    dataset: str,
    data_path: str,
    output_path: str,
    format: str = "eps",
    combine: bool = False,
):
    assert dataset in ["mnist", "cifar10"], "Dataset must be either mnist or cifar10"

    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    files = MNIST_FILES if dataset == "mnist" else CIFAR10_FILES

    dfs = [pd.read_csv(data_path / f) for f in files]

    n = 1
    if combine:
        n = len(COLUMNS)

        fig, axs = plt.subplots(
            1,
            n,
            figsize=(FIG_SIZE[0] * n, FIG_SIZE[1]),
            constrained_layout=True,
        )

        for i, col in enumerate(COLUMNS):
            axs[i].set_title(COLUMN_HUMANIZE[i])
            plot_column_from_dfs(axs[i], dfs, y=col, x="k")
            axs[i].legend().set_visible(False)

        handles, labels = axs[-1].get_legend_handles_labels()
        order = [2, 0, 1, 3, 4]
        # order = [2, 0, 1]
        # print(handles, labels)
        lgd = fig.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            bbox_to_anchor=(0.5, -0.1),
            loc="lower center",
            ncol=5,
        )
        # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # fig.legend(lines, labels)

        # fig.tight_layout()
        # fig.legend(bbox_to_anchor=(1.25, 0.6), loc="center right")

        fig.savefig(
            output_path / f"{dataset}_all.{format}",
            format=format,
            dpi=1200,
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        for col in COLUMNS:
            fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, constrained_layout=True)
            plot_column_from_dfs(ax, dfs, y=col, x="k")
            fig.savefig(
                output_path / f"{dataset}_{col}.{format}",
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
