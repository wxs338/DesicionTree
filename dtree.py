# -*- coding: utf-8 -*-

import argparse
from mldata import *
from math import log
import operator
import pickle
import random


def create5folders(dataset):
    totalnumsamples = len(dataset)
    totallist = range(0, totalnumsamples)

    random.seed(12345)
    shuffledtotallist = random.shuffle(totallist)
    firstfold = [dataset[i] for i in shuffledtotallist if i % 5 == 0]
    secondfold = [dataset[i] for i in shuffledtotallist if i % 5 == 1]
    thirdfold = [dataset[i] for i in shuffledtotallist if i % 5 == 2]
    fourthfold = [dataset[i] for i in shuffledtotallist if i % 5 == 3]
    fifthfold = [dataset[i] for i in shuffledtotallist if i % 5 == 4]

    return firstfold, secondfold, thirdfold, fourthfold, fifthfold


# entropy function
def calcEntropy(dataset):

    numsamples = len(dataset)
    labelCounts = {}
    for features in dataset:
        currentLabl = features[-1]
        if currentLabl not in labelCounts.keys():
            labelCounts[currentLabl] = 0

        labelCounts[currentLabl] += 1
    entro = 0.0
    for key in labelCounts:
        prop = float(labelCounts[key])/numsamples
        entro -= prop * log(prop, 2)
    return entro


#def createDataSet():
#    dataset = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,0,'no']]
#    labels = ['no surfacing','flippers']
#    return dataset, labels

def splitDataSet(dataSet, axis, value):

    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = list(featVec[:axis])
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def getSubDataset(dataset,colindex,value):
    subdataset = [] # 用于存储子数据集
    for rowvector in dataset:
        if rowvector[colindex] == value:
            # 下边两句实现抽取除第colIndex列特征的其他特征取值
            subrowvector = rowvector[:colindex]
            subrowvector.extend(rowvector[colindex+1:])
            # 将抽取的特征行添加到特征子数据集中
            subdataset.append(subrowvector)
    return subdataset



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

# test Gini values
print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def gainRatioNumeric(category,attributes):
    categories = []
    for i in range(len(attributes)):
        if not attributes[i] == "?":
            categories.append([float(attributes[i]),category[i]])
    categories = sorted(categories, key = lambda x:x[0])
    attri = [categories[i][0] for i in range(len(categories))]
    cate = [categories[i][1] for i in range(len(categories))]
    if len(set(attri))==1:
        return 0
    else:
        gainValues = []
        divPoint = []
        for i in range(1, len(cate)):
            if not attri[i] == attri[i-1]:
                gainValues.append(calcEntropy(cate[:i]) * float(i) / len(cate) + calcEntropy(cate[i:]) * (1-float(i) / len(cate)))
                divPoint.append(i)
        gain = calcEntropy(cate) - min(gainValues)
        pValue = float(divPoint[gainValues.index(min(gainValues))])/len(cate)
        entryAttribute = -pValue * math.log(pValue,2) - (1 - pValue) * math.log((1 - pValue), 2)
        value = gain / entryAttribute
        return value


def gainRatioNominal(category, attributes):
    attribute = []
    categories = []
    offset = 0
    for a in range(len(attributes)):
        if not attributes[a] == "?":
            attribute.append(attributes[a])
            categories.append(category[a])
    for a in set(attribute):
        categoryKind = []
        partition = float(attribute.count(a)) / len(attribute)
        for b in range(len(categories)):
            if attribute[b] == a:
                categoryKind.append(categories[b])
        offset = offset + partition * calcEntropy(categoryKind)
    entropyOfAttributes = calcEntropy(attribute)
    gain = calcEntropy(categories) - offset
    if entropyOfAttributes == 0:
        return 0
    else:
        result = gain / entropyOfAttributes
        return result

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
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

def bestFeatToGetSubdataset(dataset):
    # 下边这句实现：除去最后一列类别标签列剩余的列数即为特征个数
    numFeature = len(dataset[0]) - 2
    baseEntropy = calcEntropy(dataset)
    print("baseEntropy =", baseEntropy)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(1, numFeature+1):# i表示该函数传入的数据集中每个特征
        # 下边这句实现抽取特征i在数据集中的所有取值

        feat_i_values = [example[i] for example in dataset]
        uniqueValues = set(feat_i_values)
        feat_i_entropy = 0.0
        for value in uniqueValues:
            subDataset = splitDataSet(dataset, i, value)
            # 下边这句计算pi
            prob_i = len(subDataset)/float(len(dataset))
            feat_i_entropy += prob_i * calcEntropy(subDataset)
        infoGain_i = baseEntropy - feat_i_entropy
        # print("InfoGain  = ", infoGain_i)
        if (infoGain_i > bestInfoGain):
            bestInfoGain = infoGain_i
            bestFeature = i
            print("BestFeature", bestFeature)
    return bestFeature

def mostClass(ClassList):
    classCount = {}
    for class_i in ClassList:
        if class_i not in classCount.keys():
            classCount[class_i] = 0
        classCount[class_i] += 1
    sortedClassCount = sorted(classCount.iteritems(),
    key=operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def creatTree(dataset,features):
    classList = [example[-1] for example in dataset]
    #判断传入的dataset中是否只有一种类别，是，返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #判断是否遍历完所有的特征,是，返回个数最多的类别
    if len(dataset[0]) == 1:
        return mostClass(classList)
    #找出最好的特征划分数据集
    bestFeat = bestFeatToGetSubdataset(dataset)
    #找出最好特征对应的标签
    bestFeatLabel = features[bestFeat]
    #搭建树结构
    myTree = {bestFeatLabel:{}}
    del (features[bestFeat])
    #抽取最好特征的可能取值集合
    bestFeatValues = [example[bestFeat] for example in dataset]
    uniqueBestFeatValues = set(bestFeatValues)
    for value in uniqueBestFeatValues:
        #取出在该最好特征的value取值下的子数据集和子标签列表
        subDataset = getSubDataset(dataset, bestFeat, value)
        subLabels = features[:]
        #递归创建子树
        myTree[bestFeatLabel][value] = creatTree(subDataset, subLabels)
    return myTree

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

def storeTree(trainTree, filename):

    fw = open(filename, 'wb')
    pickle.dump(trainTree, fw)
    fw.close()

def grabTree(filename):

    fr = open(filename, 'rb')
    return pickle.load(fr)


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
    trainTree = creatTree(dataset, features)
    print(trainTree)
    storeTree(trainTree, (args.dataLocation+" Tree"))
    classlabel = classify(trainTree, storelabels, dataset[5])
    print("At the end", classlabel)
