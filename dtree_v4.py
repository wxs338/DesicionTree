

import argparse
from pathlib import Path
import os
from mldata import *
import createdtree_v2


import operator
import pickle
import random


def cross_validation_5folds(dataset):
    random.seed(12345)
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

def fiveFolderscompute(dataset, features, infogainType, max_depth):
    folds = cross_validation_5folds(dataset)
    #features_copy = features
    scores = list()
    for fold in folds:
        features_copy = features.copy()
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        if infogainType == 0:
            trainTree = createdtree_v2.create_ID3tree(train_set, features_copy, max_depth, 0)
        elif infogainType == 1:
            trainTree = createdtree_v2.create_C45tree(train_set, features_copy, max_depth, 0)
        #prodictions = decision_tree(trainTree, features_copy, test_set)
        prodictions = classifyAll(trainTree, features, test_set)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, prodictions)
        scores.append(accuracy)
    return scores



def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
            return class_label


def classifyAll(inputTree, featLabels, testDataSet):
    classLabelAll = []
    for testVec in testDataSet:
        testVec.pop(-1)
        #print(testVec)
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


if __name__ == '__main__':
    # Command line parameters
    parser = argparse.ArgumentParser()
    
    # Option 1 : path to the data
    parser.add_argument('dataLocation', help="Input your data location after Python file",
                        type=str)
    parser.add_argument('validationType', help="0 for cross validation, 1 for run algorithm on the full sample",
                        type=int, choices=range(0, 2))
    parser.add_argument('max_depth_of_tree', help="Maximum depth of the tree, 0 for full tree",
                        type=int, choices=range(0, 1000))

    parser.add_argument('informationGainType', help="0 for information gain, 1 for gain ratio",
                        type=int, choices=range(0, 2))

    args = parser.parse_args()

    print("Dateset Location:", args.dataLocation, "Validation Type:", args.validationType,
          "Max Depth:", args.max_depth_of_tree, "Info Gain Type:", args.informationGainType)

    # load dataset
    dataPath = Path(args.dataLocation)
    (dirname, dataname) = os.path.split(dataPath)

    dataset = parse_c45(os.path.basename(dataPath), rootdir = dirname)

    max_depth = args.max_depth_of_tree
    validation_type = args.validationType
    split_criterion = args.informationGainType

    features = [feature.name for feature in dataset.schema]
    features.pop(0)
    features.pop(-1)
    featurescopy = features.copy()
    print(features)
    print("The number of feature is ", len(features))

    #dataset1 = dataset[:1000]

    actual = [row[-1] for row in dataset]

    
    for example in dataset:
        example.pop(0)
    #print(dataset[10])

    # Option 2: validation_type: 0 for cross validation, 1 for full sample

    if validation_type == 0:
        scores = fiveFolderscompute(dataset, features, split_criterion, max_depth)
        if len(scores) != 0:
            average_scores = sum(scores)/len(scores)
        print(scores, average_scores)

    elif validation_type == 1:

        testsample = ['-', '+', '-', '-', '-', '+', '+', '+', '+', '+', '+']


         # Option 4: split_criterion: 0 for information gain, 1 for gain ratio

        if split_criterion == 0:
            trainTree = createdtree_v2.create_ID3tree(dataset, featurescopy, max_depth, 0)
        else:
            trainTree = createdtree_v2.create_C45tree(dataset, featurescopy, max_depth, 0)
        prodictions = classify(trainTree, featurescopy, dataset)

        accuracy = accuracy_metric(actual, prodictions)
        #print ("The test testsample is " + ','.join(testsample))
        #print ("The lable for the testsample is " + str(classlabel))








