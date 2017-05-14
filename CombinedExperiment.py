import numpy as np
import os

from keras import initializations
from weightInitialization import *
from keras_helpers import *
from evaluation import *
from plotting import *
from lib.Outcome import Outcome
from lib.utils import soundNotification
from lib.Experiment import Experiment

class CombinedExperiment():

    def __init__(self, outputPath, experiments = []):
        self.outputPath = output
        self.experiments = experiments

    def reset(self):
        for e in self.experiments:
            e.reset()

    def train(self):
        for e in self.experiments:
            e.train()

    def add(self, experiment):
        self.experiments.append(experiment)

    def plotAll(self):
        for e in self.experiments:
            for metric in e.getMetrics():
                e.plot(metric)
