import numpy as np


def initializeModels(models, initializer):

    for model in models:
        initializeModel(model, initializer)

def initializeModel(model, initializer):

    initializer = np.random.randn

    weightArchitecture = model.get_weights()

    weights = []
    for w in weightArchitecture:

        weights.append(initializer(w.shape))

    model.set_weights(model)

def normalWeightInitializer(model, mean, variance):

    weightStructure = model.get_weights()

    weights = []
    for layer in weightStructure:

        weights.append(mean + variance * np.random.randn(*layer.shape))

    model.set_weights(weights)
