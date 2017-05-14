from keras.models import Sequential
from keras.layers.core import Dense, Activation


def feedForwardExperiment(nModel, architecture, inputSize, nClasse):

    models = []
    for i in range(nModel):
        models.append(feedForward(architecture, inputSize, nClasse))

    return models

def feedForward(architecture, dataParameters, softmax=False):

    nClasses = dataParameters['nClass']
    inputSize = dataParameters['inputSize']

    model = Sequential()
    model.add(Dense(output_dim=architecture[0], input_dim=inputSize))

    for iLayer in range(1, len(architecture)):

        layer = architecture[iLayer]

        model.add(Dense(output_dim=layer, activation="relu"))

    if softmax:
        model.add(Dense(output_dim=nClasses, activation="softmax"))
    else:
        model.add(Dense(output_dim=nClasses, activation="relu"))

    return model


def compileModels(models, optimizer, loss):

    for model in models:
        model.compile(optimizer, loss)



def trainModels(models, nb_epoch, batch_size, X, Y, verbose=0, callbacks=[]):

    for i, model in enumerate(models):
        print('Training model {}\r'.format(i))
        model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=callbacks)
