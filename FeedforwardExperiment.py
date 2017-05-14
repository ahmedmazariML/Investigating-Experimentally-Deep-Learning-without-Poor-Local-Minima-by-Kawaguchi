import numpy as np
import os

from keras import initializations
from weightInitialization import *
from keras_helpers import *
from evaluation import *
from plotting import *
from Outcome import *
from lib.utils import soundNotification

class Experiment(object):

    def __init__(self, name, parameters):

        self.parameters = parameters
        self.parameterLabels = {}
        self.name = name

    @staticmethod
    def loadFromDisk():
        pass

    def getName(self):
        return self.name

    def setParameter(self, name, value, label=None):
        self.parameters[name] = value
        self.parameterLabels[name] = label

    def setParameters(self, parameters):

        self.parameters.update(parameters)

    def useParameterForName(self, name, value):
        pass

    def getOutcomes(self):
        return self.outcomes

    def beforeRun(self):
        pass

    def afterRun(self):
        pass

    def reset(self):
        pass

    def run(self):
        raise Exception('Should be overriden')

    def saveToDisk(self):
        pass

    def notifyEnd(self):
        soundNotification()


class FeedforwardExperiment(Experiment):

    def __init__(self, experimentName, parameters, dataParameters, prepareExperiment=False):

        # super().__init__(experimentName, parameters)
        super(FeedforwardExperiment, self).__init__(experimentName, parameters)

        self.dataParameters = dataParameters

        if (prepareExperiment):
            self.prepare()

    def prepare(self):

        self._buildModels()
        self.reset()
        self._compileModels()

    def initializeWeights(self, strategy, params):

        for i, model in enumerate(self.models):
            strategy(model, **params[i])

    def reset(self):

        self.outcomes = {

            'weightDiffs': PairModelOutcome(self, "Weight difference above threshold", "Epoch", "N"),
            'trainEvaluations': MultipleModelOutcome(self, "Train error", "Epoch", "Error", self.experimentNames),
            'testEvaluations': MultipleModelOutcome(self, "Test error", "Epoch", "Error", self.experimentNames),

            'trainLabelMatchs': PairModelOutcome(self, "Number of label identically classfied (train)", "Epoch", "N"),
            'testLabelMatchs': PairModelOutcome(self, "Number of label identically classfied (test)", "Epoch", "N"),

            'trainLabelMismatchs': PairModelOutcome(self, "Number of label not identically classfied (train)", "Epoch", "N"),
            'testLabelMismatchs': PairModelOutcome(self, "Number of label not identically classfied (test)", "Epoch", "N"),
            'weights': []
        }


    def setValidationData(self, X_test, Y_test):
        self.X_test = X_test
        self.Y_test = Y_test

    def run(self, X_train, Y_train, weightDiffThreshold = 0.01, nMetaEpoch = 20, nEpoch = 5, batch_size = 64):

        if (self.X_test is None or self.Y_test is None):
            raise Exception('set validation data with setValidationData method')
        self.X_train = X_train
        self.Y_train = Y_train

        #print(nEpoch, batch_size, metaEpoch)
        for i in range(nMetaEpoch):
            print("epoch {}".format((i + 1) * nEpoch))

            trainModels(self.models, nEpoch, batch_size, X_train, Y_train)
            self._collectMetrics()


    def _buildModels(self):
        params = self.parameters
        architecture = params['architecture']
        nModel = params['nModel']        
        softmax = params['softmax']
        models = []
        self.experimentNames = []
        for i in range(nModel):
            models.append(feedForward(architecture, self.dataParameters, softmax))
            self.experimentNames.append("model {} ({})".format(i, "_".join([str(x) for x in architecture])))

        self.models = models


    def _compileModels(self):

        optimizer = self.parameters['optimizer']
        loss = self.parameters['loss']
        compileModels(self.models, optimizer, loss)

    def _initializeAllModelWeighs(self, strategy):

        for model in self.models:
            initializeModel(model, strategy)

    def _collectMetrics(self):

        X_train = self.X_train
        Y_train = self.Y_train

        X_test = self.X_test
        Y_test = self.Y_test

        models = self.models
        outcomes = self.outcomes

        print('computing outcomes')

        self.outcomes['weights'].append(self.getAllModelWeights())
        evaluateModels(models, X_train, Y_train, outcomes['trainEvaluations'])
        evaluateModels(models, X_test, Y_test, outcomes['testEvaluations'])

        getLabelHavingSameClassification(models, X_train, Y_train, outcomes['trainLabelMatchs'], outcomes['trainLabelMismatchs'])
        getLabelHavingSameClassification(models, X_test, Y_test, outcomes['testLabelMatchs'], outcomes['testLabelMismatchs'])
        weightDiff(models, self.outcomes['weightDiffs'])


    def getAllModelWeights(self):
        modelsWeights = []
        for model in self.models:
            modelsWeights.append(model.get_weights())

        return modelsWeights

    def plotAllOutcomes(self, save=True):
        for outcome in self.outcomes.values():
            if isinstance(outcome, Outcome):
                outcome.plot(save)
