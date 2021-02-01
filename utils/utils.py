import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

##CODE TO READ MNIST DATASET FROM: https://github.com/WHDY/mnist_cnn_numba_cuda/blob/b7ee58d072aa253ad248d134d53ce2b4cfb155e6/read_mnist.py#L74
import math

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data
    
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]
def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        # return dense_to_one_hot(labels)
        return labels

def plot_image(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title("Full size image:")

def load_data(path = None):
    if path: data_dir = path
    else: data_dir = 'mnist'
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    train_images = extract_images(train_images_path)
    train_labels = extract_labels(train_labels_path)
    test_images = extract_images(test_images_path)
    test_labels = extract_labels(test_labels_path)
    
    train_data_size = train_images.shape[0]
    test_data_size = test_images.shape[0]
    
    train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

    train_data_size = train_images.shape[0]
    test_data_size = test_images.shape[0]

    #NORMALIZE
    train_images = train_images.astype(np.float32)
    train_images = np.multiply(train_images, 1.0 / 255.0)
    test_images = test_images.astype(np.float32)
    test_images = np.multiply(test_images, 1.0 / 255.0)

    #RANDOMIZE THE TRAIN DATA ORDER
    order = np.arange(train_data_size)
    np.random.shuffle(order)

    #PICK TRAIN & TEST DATA
    train_data = train_images[order]
    train_label = train_labels[order]

    test_data = test_images
    test_label = test_labels

    #RESHAPE DATA
    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)
    return train_data,train_label, test_data, test_label

def next_batch(train_data, train_label, batch_size):
    order = np.arange(train_data.shape[0])
    np.random.shuffle(order)
    train_data_batch = train_data[order]
    train_label_batch = train_label[order]
    start = 0
    end = batch_size
    return train_data_batch[0: batch_size], train_label_batch[0: batch_size]
