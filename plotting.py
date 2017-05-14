import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plotOutcomeObjects(outcomes, exportName):
    pass


def plotOutcomes(outcomes, outcomeNames, exportName, label_x, label_y):

    cwd = os.getcwd()
    output = os.path.join(cwd, "..", "report", "images", "")

    plt.figure()

    for i in range(len(outcomes)):
        outcome = np.array(outcomes[i])
        plt.plot(range(len(outcome)), outcome, label=outcomeNames[i])


    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output, "{}.png".format(exportName)))

def plotMetrics(nModel, trainEvaluations, testEvaluations, trainLabelMatchs, trainLabelMismatchs, testLabelMatchs, testLabelMismatchs, weightDiffs, weightDiffThreshold, experimentName):


    nMetaEpoch = len(trainEvaluations)
    print("nMetaEpoch: {}", nMetaEpoch)

    cwd = os.getcwd()
    output = os.path.join(cwd, "..", "report", "images", "")
    trainEvaluations = np.array(trainEvaluations)
    testEvaluations = np.array(testEvaluations)

    trainLabelMatchs = np.array(trainLabelMatchs)
    trainLabelMismatchs = np.array(trainLabelMismatchs)

    testLabelMatchs = np.array(testLabelMatchs)
    testLabelMismatchs = np.array(testLabelMismatchs)

    weightDiffs = np.array(weightDiffs)
    plt.figure()
    print('Training error')
    for i in range(nModel):
        plt.plot(trainEvaluations[:, i])
        plt.savefig(os.path.join(output, experimentName + "_train_evaluation.png"))
    plt.show()

    plt.figure()
    print('Test error')
    for i in range(nModel):
        plt.plot(testEvaluations[:, i])
        plt.savefig(os.path.join(output, experimentName + "_test_evaluation.png"))
    plt.show()

    plt.figure()
    print('Number of identically predicted examples (training set)')
    for i in range(nModel - 1):
        plt.plot(trainLabelMatchs[:, i])
        # plt.plot(trainLabelMismatchs[:, i])
        # plt.plot(trainLabelMismatchs[:, i] + trainLabelMatchs[:, i])
        plt.savefig(output + experimentName + "_train_predictions_match_mismatch.png")
    plt.show()

    # plt.figure()
    # plt.title('Number of different predicted examples (training set)')
    # for i in range(nModel -1):
    #     plt.plot(trainLabelMismatchs[:, i])
    #     plt.savefig(output + "different_train_predictions.png")
    # plt.show()

    plt.figure()
    print('Number of identically predicted examples (test set)')
    for i in range(nModel - 1):
        plt.plot(testLabelMatchs[:, i])
        # plt.plot(testLabelMismatchs[:, i])
        # plt.plot(testLabelMismatchs[:, i] + testLabelMatchs[:, i])
        plt.savefig(output + experimentName + "_test_predictions_match_mismatch.png")
    plt.show()

    # plt.figure()
    # plt.title('Number of different predicted examples (test set)')
    # for i in range(nModel -1):
    #
    #     plt.savefig(output + "different_test_predictions.png")
    # plt.show()

    plt.figure()
    print('Number of weights differences being above {}'.format(weightDiffThreshold))
    for i in range(nModel -1):
        plt.plot(weightDiffs[:, i])
        plt.savefig(output + experimentName + "_weight_differences.png")
    plt.show()
