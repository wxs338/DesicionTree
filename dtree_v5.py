
import argparse
from pathlib import Path
import os
from mldata import *
import createdtree

from math import log
import operator
import pickle



# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
	file = open(filename, "rb")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
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
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
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

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Calculate the entropy for a split dataset
def calcShannonEnt(dataSet, classvalues):

    numberData=len(dataSet)
    classCount={}
    shannonEnt=0

    for one in dataSet:
        currentLabel=one[-1]
        if currentLabel not in classCount.keys():
            classCount[currentLabel]=0
        classCount[currentLabel]+=1
    for key in classCount:
        prob=float(classCount[key])/numberData
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 100, 100, 100, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            entropy = calcShannonEnt(groups, class_values)
            if entropy < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], entropy, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}
'''  
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:            
			groups = test_split(index, row[index], dataset)
			# gini = gini_index(groups, class_values)
            entropy = calcShannonEnt(groups, class_values)
            if entropy < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], entropy, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}
'''
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

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
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)



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

    dataset1 = dataset[:][1:]
    print(dataset[0])
    print(dataset1[0])

    # Test CART on Bank Note dataset
    seed(12345)
    # load and prepare data
    #filename = 'data_banknote_authentication.csv'
    #dataset = load_csv(filename)
    # convert string attributes to integers
    #for i in range(len(dataset[0])):
    #    str_column_to_float(dataset, i)
    # evaluate algorithm
    n_folds = 5
    max_depth = 5
    min_size = 1
    scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

    #print("The Dataset is", args.dataLocation)
    #print("The number of examples in Dataset is ", len(dataset))

    features = [feature.name for feature in dataset.schema]


    print("The number of feature is ", len(dataset.schema))

    print("Features are: ", features)

    featurescopy = features[1:-1]  # copy features
    print(featurescopy)

    # Option 2 : 0 for cross validation, 1 for full sample
"""
    if validation_type == 0:
        scores = fiveFolderscompute(dataset[0:20], featurescopy, split_criterion, max_depth)

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

"""