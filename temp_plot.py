from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

filepaths = ["mnist_pruned_single_layer.csv", "mnist_non_pruned_single_layer.csv"]

for filepath in filepaths:
    df = pd.read_csv(filepath)
    path = Path(filepath)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

    df.plot(
        x="k", y="bc_persisted_size", ax=ax, c="green", linestyle="--", label="Original"
    )
    df.plot(x="k", y="ac_persisted_size", ax=ax, c="maroon", label="Compressed")

    ax.set_xlabel("k")
    ax.set_ylabel("Persisted Size (in KB)")
    ax.set_title("MNIST Pruned Single Layer")

    fig.savefig(f"{path.stem}.png", dpi=1200)
    plt.close(fig)
