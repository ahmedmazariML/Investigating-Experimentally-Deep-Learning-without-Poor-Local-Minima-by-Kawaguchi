import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from keras.optimizers import SGD
from models import *
from experiments import *
from evaluation import *

import matplotlib.pyplot as plt


nb_classes = 10
# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class bmatrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

nModel = 2


callbacks = []

loss="mean_squared_error"
optimizer = SGD(lr=learningRate)
models = feedForwardExperiment(nModel)

compileModels(models, optimizer, loss)

initializeWeights(models)

trainEvaluations = []
testEvaluations = []
labelMatchs = []
labelMismatchs = []
weightDiffs = []


metaEpoch = 2
nEpoch = 2
batch_size = 64

#print(nEpoch, batch_size, metaEpoch)
for i in range(metaEpoch):
    print("epoch {}".format((i +1) * nEpoch))

    trainModels(models, nEpoch, batch_size, X_train, Y_train)

    nSameLabel, nDifferentLabel = getLabelHavingSameClassification(models, X_train, Y_train)
    labelMatchs.append(nSameLabel)
    labelMismatchs.append(nDifferentLabel)

    weightDiffs.append(weightDiff(models))

    trainEvaluations.append(evaluateModels(models, X_train, Y_train))
    testEvaluations.append(evaluateModels(models, X_test, Y_test))
            
trainEvaluations = np.array(trainEvaluations)
testEvaluations = np.array(testEvaluations)

labelMatchs = np.array(labelMatchs)
labelMismatchs = np.array(labelMismatchs)
