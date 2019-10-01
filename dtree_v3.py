

import argparse
from mldata import *
import id3
import c45

import operator
import pickle
import random


def cross_validation_5folds(dataset):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / 5)
    for i in range(5):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, *args):
    folds = cross_validation_5folds(dataset)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = id3.build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)

def classify(inputTree,featlabels,testFeatValue):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featlabels.index(firstStr)
    classLabel = None
    for firstStr_value in secondDict.keys():
        if testFeatValue[featIndex] == firstStr_value:
            if type(secondDict[firstStr_value]).__name__ == 'dict':
                classLabel = classify(secondDict[firstStr_value], featlabels, testFeatValue)
            else:
                classLabel = secondDict[firstStr_value]
    return classLabel


if __name__ == '__main__':
    # Command line parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('dataLocation', help="Input your data location after Python file",
                        type=str)
    parser.add_argument('validationType', help="0 for cross validation, 1 for run algorithm on the full sample",
                        type=int, choices=range(0, 2))
    parser.add_argument('max_depth_of_tree', help="Maximum depth of the tree, 0 for full tree",
                        type=int, choices=range(0, 1000))

    parser.add_argument('informationGainType', help="0 for information gain, 1 for gain ratio",
                        type=int, choices=range(0, 2))

    args = parser.parse_args()

    print("Here you go, input options on CMD: ", "Datesetname ", args.dataLocation, "Validation Type ", args.validationType,
          "Max Depth ", args.max_depth_of_tree, "Info Gain Type ", args.informationGainType)

    # load dataset

    dataset = parse_c45(args.dataLocation)

    print("The Dataset is", args.dataLocation)
    print("The number of examples in Dataset is ", len(dataset))

    features = [feature.name for feature in dataset.schema]

    print("The number of feature is ", len(dataset.schema))

    print("Features are: ", features)

    print("Looks Load Dataset Sucessfully!!")

    # 5 folder divide

    storelabels = features[:]#复制label
    trainTree = id3.CreateTree(dataset, features)
    print(trainTree)
    id3.storeTree(trainTree, (args.dataLocation+" Tree"))
    classlabel = classify(trainTree, storelabels, dataset[5])
    print("At the end", classlabel)