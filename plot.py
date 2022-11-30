import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

COLUMNS = ["test_acc", "test_time", "mem_size", "persisted_size"]
FIG_SIZE = (4, 3)


@ticker.FuncFormatter
def second_to_ms_formatter(x, pos):
    return f"{x * 1000:g} ms"


def plot_column(df: pd.DataFrame, y: str, x: str = "k"):
    fig, ax = plt.subplots(
        1, 1, figsize=FIG_SIZE, constrained_layout=True
    )
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
        ax.yaxis.set_major_formatter(
            ticker.StrMethodFormatter("{x:g} KB")
        )
        # ax.set_ylabel("in KB")
    if "time" in y:
        ax.yaxis.set_major_formatter(second_to_ms_formatter)
        # ax.set_ylabel("in ms")
        # ax.set_ylabel("Size in (KB)")

    ax.xaxis.set_major_formatter(
        ticker.ScalarFormatter(useMathText=True)
    )
    # fig.suptitle(title)
    # fig.tight_layout()
    # ax.legend().set_visible(False)
    return fig


def plot(
    csv: Union[str, Path], output_path: str, format: str = "eps"
):
    dataset = "mnist" if "mnist" in csv else "cifar10"
    # output_path = BASE_OUTPUT_PATH
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(csv)

    for col in COLUMNS:
        fig = plot_column(df, col, x="k")
        fig.savefig(
            output_path / f"{dataset}_{col}.{format}",
            format=format,
            dpi=1200,
        )
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="mnist.csv", type=str)
    parser.add_argument("-o", "--output", default="./plots", type=str)
    parser.add_argument("--format", default="eps", type=str)
    args = parser.parse_args()
    plot(args.file, args.output, format=args.format)
