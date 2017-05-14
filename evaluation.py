import numpy as np
import keras

def trainingError():
    pass


def getLabelHavingSameClassification(models, X, Y, sameOutcome, differentOutcome):

    predictions = []

    for model in models:
        predictions.append(model.predict_classes(X, verbose = 0).astype(int))

    for i in range(len(models)):
        for j in range(i + 1, len(models)):

            diff = predictions[i] - predictions[j]

            nIdenticals = np.sum(diff == 0)
            nDifferents = np.sum(diff != 0)

            sameOutcome.collect(i, j, nIdenticals)
            differentOutcome.collect(i, j, nDifferents)


def evaluateModels(models, X, Y, collector):

    predictions = []
    for i, model in enumerate(models):

        collector.collect(i, model.evaluate(X, Y, verbose=0))

def weightDiff(models, outcomeCollector, diffThreshold = 0.1):

    weights = []
    for model in models:

        weights.append(model.get_weights())

    layerDifferences = {}

    for iModel in range(len(models)):
        for jModel in range(iModel + 1, len(models)):

            modelLayerDifferences = 0

            for layer in range(len(weights)):

                diff = weights[iModel][layer] - weights[jModel][layer]

                modelLayerDifferences += np.sum(diff > diffThreshold)

            outcomeCollector.collect(iModel, jModel, modelLayerDifferences)
