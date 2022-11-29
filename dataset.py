import gzip
import hashlib
import os
import pickle
import sys
import tarfile
import urllib
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
from tqdm import tqdm

# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
USER_AGENT = "pytorch/vision"


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == _calculate_md5(fpath, **kwargs)


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _urlretrieve(
    url: str, filename: str, chunk_size: int = 1024 * 32
) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT}
        )
    ) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            filename,
            length=response.length,
        )


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers)
        ) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )


# taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
def _download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    max_redirect_hops: int = 3,
):
    fpath = os.path.expanduser(root)
    if filename:
        fpath = os.path.join(root, filename)

    url = _get_redirect_url(url, max_hops=max_redirect_hops)

    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, OSError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead. Downloading "
                + url
                + " to "
                + fpath
            )
            _urlretrieve(url, fpath)
        else:
            raise e

    if not _check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def mnist(path, one_hot=False):
    """
    return: train_set, valid_set, test_set
    train_set size: (50000, 784), (50000,)
    valid_set size: (10000, 784), (10000,)
    test_set size: (10000, 784), (10000,)
    feature: numerical in range [0, 1]
    target: categorical from 0 to 9
    """

    URL = "https://figshare.com/ndownloader/files/25635053"
    MD5_SUM = "a02cd19f81d51c426d7ca14024243ce9"

    if not _check_integrity(path, MD5_SUM):
        print("Downloading MNIST dataset...")
        _download_url(URL, root=str(path), md5=MD5_SUM)

    if not _check_integrity(path, MD5_SUM):
        raise RuntimeError("Dataset not found or corrupted.")

    # load the dataset
    with gzip.open(path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(
            f, encoding="latin1"
        )

    def get_one_hot(targets, nb_classes):
        return np.eye(nb_classes)[np.array(targets).reshape(-1)]

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        valid_set = (valid_set[0], get_one_hot(valid_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))

    return train_set, valid_set, test_set


def cifar10(path: Union[str, Path], one_hot=False):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Downloaded from: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

    Args:
        path (str): Directory containing CIFAR-10.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    files = [
        "cifar-10-batches-bin/data_batch_1.bin",
        "cifar-10-batches-bin/data_batch_2.bin",
        "cifar-10-batches-bin/data_batch_3.bin",
        "cifar-10-batches-bin/data_batch_4.bin",
        "cifar-10-batches-bin/data_batch_5.bin",
        "cifar-10-batches-bin/test_batch.bin",
    ]

    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    MD5_SUM = "c32a1d4ab5d03f1284b67883e8d87530"

    if not _check_integrity(path, MD5_SUM):
        print("Downloading CIFAR-10 dataset...")
        _download_url(URL, root=str(path), md5=MD5_SUM)

    if not _check_integrity(path, MD5_SUM):
        raise RuntimeError("Dataset not found or corrupted.")

    # Load data from tarfile
    with tarfile.open(path) as tar_object:
        # print([file for file in tar_object])

        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype="uint8")

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize : (i + 1) * fsize] = np.frombuffer(
                f.read(), "B"
            )

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072).astype("float32") / 255

    # Split into train and test
    train_images, valid_images, test_images = (
        images[:50000],
        images[:10000],
        images[50000:],
    )
    train_labels, valid_labels, test_labels = (
        labels[:50000],
        labels[:10000],
        labels[50000:],
    )

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype="uint8")
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    if one_hot:
        train_set = (train_images, _onehot(train_labels), 10)
        valid_set = (valid_images, _onehot(valid_labels), 10)
        test_set = (test_images, _onehot(test_labels), 10)
    else:
        train_set = (train_images, train_labels)
        valid_set = (valid_images, valid_labels)
        test_set = (test_images, test_labels)
    return train_set, valid_set, test_set
