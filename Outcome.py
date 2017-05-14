import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

class Outcome():

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def newX(self, x=None):
        if not x:
            self.x.append(len(self.x) + 1)
        else:
            self.x.append(x)

    def setName(self, experiment, outcomeName):

        paramNames = []

        for param in experiment.parameters:

            if param in experiment.parameterLabels.keys():
                paramNames.append(experiment.parameterLabels[param])
            else:
                paramNames.append(experiment.parameters[param])

        paramNames = "_".join([str(param) for param in paramNames])
        name =  experiment.name + "_" + paramNames + "_" + outcomeName
        name = re.sub(' |, ', "_", name)
        name = re.sub('\[|\]', "", name)
        self.name = name

    def saveFigure(self, fig):

        name = re.sub(" ", "_", self.name)
        cwd = os.getcwd()
        output = os.path.join(cwd, "..", "report", "images", "")
        fig.savefig(os.path.join(output, "{}.png".format(self.name)))


class weightDistributionOutcome(Outcome):

    def collect(self):

        self.plot(epoch)

    def plot(self, epoch):
        models = self.models
        self.layerByLayerWeightDistribution(models)
        self.allLayerWeightDistributions(models)


    def layerByLayerWeightDistribution(models):

        for layer in range(len(models[0].get_weights())):
            f = plt.figure()
            for i, model in enumerate(models):
                modelWeights = model.get_weights()[layer].reshape(-1, 1)
                plt.hist(modelWeights, bins=50, label="model {}".format(i))
                plt.legend()
            self.saveFigure(f)

    def allLayerWeightDistributions(models):

        for model in models:
            plt.figure()
            for layer in range(len(model.get_weights())):
                modelWeights = model.get_weights()[layer].reshape(-1, 1)
                plt.hist(modelWeights, bins=50, label="layer {}".format(layer))
                plt.legend()


class PairModelOutcome(Outcome):

    def __init__(self, experiment, name, xlabel, ylabel, labels = []):

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.setName(experiment, name)
        self.values = {}
        self.x = []
        self.labels = labels

    def collect(self, idxFirstModel, idxSecondModel, value):

        key = "{}_{}".format(idxFirstModel, idxSecondModel)

        if key in self.values:
            self.values[key].append(value)
        else:
            self.values[key] = [value]

        if (idxFirstModel == 0 and idxSecondModel == 1):
            self.newX()

    def plot(self, saveFig=False):


        f = plt.figure()
        print(self.name)

        for i in self.values:

            outcome = np.array(self.values[i])
            plt.plot(self.x, outcome, label=i)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.show()

        if (saveFig):
            self.saveFigure(f)

class MultipleModelOutcome(Outcome):

    def __init__(self, experiment, name, xlabel, ylabel, labels = []):

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.setName(experiment, name)
        self.values = {}
        self.x = []
        self.labels = labels

    def collect(self, iModel, value, x = None):

        if iModel in self.values:
            self.values[iModel].append(value)
        else:
            self.values[iModel] = [value]

        if iModel == 0:
            self.newX(x)

    def plot(self, saveFig=False):

        f = plt.figure()
        print(self.name)

        for i in self.values:

            outcome = np.array(self.values[i])
            plt.plot(self.x, outcome, label=self.labels[i])

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.show()

        if (saveFig):
            self.saveFigure(f)
