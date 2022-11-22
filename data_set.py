import gzip
import pickle
from mlp import MLP
import numpy as np
import os
import tarfile

def mnist(path, one_hot=False):
    """
    return: train_set, valid_set, test_set
    train_set size: (50000, 784), (50000,)
    valid_set size: (10000, 784), (10000,)
    test_set size: (10000, 784), (10000,)
    feature: numerical in range [0, 1]
    target: categorical from 0 to 9
    """
    
    # load the dataset
    with gzip.open(path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    def get_one_hot(targets, nb_classes):
        return np.eye(nb_classes)[np.array(targets).reshape(-1)]

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        valid_set = (valid_set[0], get_one_hot(valid_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))

    return train_set, valid_set, test_set


def cifar10(path=None,one_hot=False):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']


    # Load data from tarfile
    with tarfile.open(path) as tar_object:
        #print([file for file in tar_object])

        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype='uint8')

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
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]
    

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072).astype('float32') / 255

    # Split into train and test
    train_images,valid_images, test_images = images[:50000],images[:10000], images[50000:]
    train_labels,valid_labels, test_labels = labels[:50000],labels[:10000], labels[50000:]

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
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
    return train_set,valid_set,test_set


