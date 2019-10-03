

import argparse
from pathlib import Path
import os
from mldata import *
import createdtree


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

def fivecrossvalidationcompute(dataset): # , algorithm, n_folds, *args):
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
		predicted = decision_tree(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def fiveFolderscompute(dataset, features, infogainType, max_depth):
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
        if infogainType == 0:
            trainTree = createdtree.CreateID3Tree(dataset, features, max_depth)
        elif infogainType == 1:
            trainTree = createdtree.CreateC45Tree(train_set, features, max_depth)
        prodictions = decision_tree(trainTree, features, test_set)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, prodictions)
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
def decision_tree(tree, featlabels, testset): #, max_depth, min_size):
    #tree = createdtree.build_tree(train, max_depth, min_size)
    predictions = list()
    feature_length = len(testset[0])-2
    for row in testset:
        prediction = classify(tree, featlabels, row[1:feature_length])
        predictions.append(prediction)
    return predictions

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

def classifyAll(inputTree, featLabels, testDataSet):
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


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
    dataPath = Path(args.dataLocation)
    (dirname, dataname) = os.path.split(dataPath)

    dataset = parse_c45(os.path.basename(dataPath), rootdir = dirname)
    max_depth = args.max_depth_of_tree
    validation_type = args.validationType
    split_criterion = args.informationGainType

    #print("The Dataset is", args.dataLocation)
    #print("The number of examples in Dataset is ", len(dataset))

    features = [feature.name for feature in dataset.schema]


    print("The number of feature is ", len(dataset.schema))

    print("Features are: ", features)

    featurescopy = features[1:-1]  # copy features
    print(featurescopy)

    # Option 2 : 0 for cross validation, 1 for full sample

    if validation_type == 0:
        scores = fiveFolderscompute(dataset, featurescopy, split_criterion, max_depth)

        print(scores)

    elif validation_type == 1:
        if split_criterion == 10:
            trainTree = createdtree.CreateID3Tree(dataset, features, max_depth)
            print(trainTree)
            createdtree.storeTree(trainTree, (dataname + " Tree"))
            classlabel = classify(trainTree, featurescopy, dataset[5])
            print("At the end", classlabel)

        elif split_criterion == 11:
            trainTree = createdtree.CreateC45Tree(dataset, features, max_depth)
            print(trainTree)
            createdtree.storeTree(trainTree, (dataname + " Tree"))
            classlabel = classify(trainTree, featurescopy, dataset[5])
            print("At the end", classlabel)


    # Option 4 split criterion: 0 for information gain 1 for gain ratio
'''
    

'''







