from sklearn import datasets
from sklearn.utils import shuffle
import keras.datasets
from keras.utils import np_utils
import numpy as np

class Dataset():

    def getTrainingData(self):

        return self.X_train, self.Y_train

    def getTestData(self):

        return self.X_test, self.Y_test

    def getDataParameters(self):

        return self.dataParameters

class Cifar10Dataset(Dataset):

    def __init__(self):

        nb_classes = 10

        (X_train, y_train), (X_test, y_test) =  keras.datasets.cifar10.load_data()

        inputSize =  X_train.shape[1] ** 2
        X_train = np.mean(X_train, axis=3).reshape(X_train.shape[0], inputSize)
        X_test = np.mean(X_test, axis=3).reshape(X_test.shape[0], inputSize)

        # convert class vectors to binary class bmatrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        self.X_train = X_train
        self.Y_train = Y_train

        self.X_test = X_test
        self.Y_test = Y_test

        self.dataParameters = {
            'nClass': nb_classes,
            'inputSize': inputSize
        }


class BostonDataset(Dataset):

    def __init__(self):

        boston = datasets.load_boston()
        X, y = shuffle(boston.data, boston.target, random_state=13)
        X = X.astype(np.float32)
        offset = int(X.shape[0] * 0.9)
        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]

        self.X_train = X_train
        self.Y_train = y_train

        self.X_test = X_test
        self.Y_test = y_test

        self.dataParameters = {
            'inputSize': X.shape[1],
            'nClass': 1
        }
