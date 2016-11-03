'''
this script prepare the image dataset
read, preprocessing
mnist cifar10
'''
import os
import tarfile
import urllib
import pickle
import glob
import numpy as np
def read_mnist():
    '''
    Return train, valid, test set
    X:train.images, size: 55000*784, type: np.array
    y:train.labels, size: 55000*10, type: np.array
    valid size: 5000
    test size : 10000
    '''
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    return mnist.train, mnist.validation, mnist.test

def read_cifar10():
    '''
    Return train, valid, test set
    If cifar dataset is not downloaded, download it into current file
    X:train["images"], size: 45000*3072, type:np.array
    y:train["labels"], size: 45000*1, type:np.array
    valid size: 5000
    test size: 10000
    '''

    CIFAR_DOWNLOAD = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    CIFAR_FILE = "cifar-10-python.tar.gz"
    CIFAR_DIR = "cifar-10-batches-py"
    if not os.path.isfile(CIFAR_FILE) and not os.path.isdir(CIFAR_DIR):
        print("Downloading cifar dataset...")
        urllib.urlretrieve(CIFAR_DOWNLOAD, CIFAR_FILE)
        print("Unzipping...")
        fname = "cifar-10-python.tar.gz"
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
    all_train_image = np.array([])
    all_train_label = np.array([])
    for fname in glob.glob("cifar-10-batches-py/data_batch*"):
        f = open(fname, "rb")
        batch_dict = pickle.load(f)
        if all_train_image.shape == (0,):
            all_train_image = batch_dict["data"]
            all_train_label = batch_dict["labels"]
        else:
            all_train_image = np.concatenate((batch_dict["data"],\
                              all_train_image), axis = 0)
            all_train_label = np.concatenate((batch_dict["labels"], \
                              all_train_label), axis = 0)
        f.close()

    all_test_image = np.array([])
    all_test_label = np.array([])
    f = open("cifar-10-batches-py/test_batch")
    batch_dict = pickle.load(f)
    if all_test_image.shape == (0,):
        all_test_image = batch_dict["data"]
        all_test_label = batch_dict["labels"]
    else:
        all_test_image = np.concatenate((batch_dict["data"],\
                          all_train_image), axis = 0)
        all_test_label = np.concatenate((batch_dict["labels"], \
                          all_train_label), axis = 0)
    f.close()
    assert(all_train_image.shape==(50000,3072))
    assert(all_test_image.shape==(10000,3072))
    train={"images":all_train_image[:45000], "labels":all_train_label[:45000]}
    valid={"images":all_train_image[45000:], "labels":all_train_label[45000:]}
    test ={"images":all_test_image, "labels":all_test_image}
    return train, valid, test
