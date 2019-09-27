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



def calcShannonEnt(dataset):
    numSamples = len(dataset)
    labelCounts = {}
    for allFeatureVector in dataset:
        currentLabel = allFeatureVector[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    entropy = 0.0
    for key in labelCounts:
        property = float(labelCounts[key])/numSamples
        entropy -= property * log(property,2)
    return entropy


#def createDataSet():
#    dataset = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,0,'no']]
#    labels = ['no surfacing','flippers']
#    return dataset, labels


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

def bestFeatToGetSubdataset(dataset):
    # 下边这句实现：除去最后一列类别标签列剩余的列数即为特征个数
    numFeature = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    print("baseEntropy =", baseEntropy)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeature):# i表示该函数传入的数据集中每个特征
        # 下边这句实现抽取特征i在数据集中的所有取值
        print("A",i+1)
        feat_i_values = [example[i] for example in dataset]
        uniqueValues = set(feat_i_values)
        feat_i_entropy = 0.0
        for value in uniqueValues:
            subDataset = getSubDataset(dataset,i,value)
            # 下边这句计算pi
            prob_i = len(subDataset)/float(len(dataset))
            feat_i_entropy += prob_i * calcShannonEnt(subDataset)
        infoGain_i = baseEntropy - feat_i_entropy
        print("InfoGain  = ", infoGain_i)
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
    key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

def creatTree(dataset,labels):
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
    bestFeatLabel = labels[bestFeat]
    #搭建树结构
    myTree = {bestFeatLabel:{}}
    del (labels[bestFeat])
    #抽取最好特征的可能取值集合
    bestFeatValues = [example[bestFeat] for example in dataset]
    uniqueBestFeatValues = set(bestFeatValues)
    for value in uniqueBestFeatValues:
        #取出在该最好特征的value取值下的子数据集和子标签列表
        subDataset = getSubDataset(dataset,bestFeat,value)
        subLabels = labels[:]
        #递归创建子树
        myTree[bestFeatLabel][value] = creatTree(subDataset,subLabels)
    return myTree

def classify(inputTree,featlabels,testFeatValue):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featlabels.index(firstStr)
    for firstStr_value in secondDict.keys():
        if testFeatValue[featIndex] == firstStr_value:
            if type(secondDict[firstStr_value]).__name__ == 'dict':
                classLabel = classify(secondDict[firstStr_value],featlabels,testFeatValue)
            else: classLabel = secondDict[firstStr_value]
    return classLabel

def storeTree(trainTree,filename):

    fw = open(filename,'w')
    pickle.dump(trainTree,fw)
    fw.close()
def grabTree(filename):

    fr = open(filename)
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

    print("The number of feature is ", len(dataset.schema), "Test if list", dataset.schema[2].name)

    labels = [feature.name for feature in dataset.schema]

    print("Features are: ", labels)

    print("Looks Load Dataset Sucessfully")

    # 5 folder divide





    storelabels = labels[:]#复制label
    trainTree = creatTree(dataset, labels)
    print(trainTree)
    classlabel = classify(trainTree, storelabels, [0, 1])
    print("At the end", classlabel)
