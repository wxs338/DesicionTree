
from math import log
import operator
import pickle

#创建数据集
def createDataSet():
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发', '声音']  # 两个特征
    return dataSet, labels

#从当前许多类别中挑出数目最多的类别
def majorityClass(classList):
    list={}
    for one in classList:
        if one not in list.keys():
            list[one]=0
        list[one]+=1
    sortedClass=sorted(list.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClass[0][0]

#计算给定数据集的信息熵
def calcShannonEnt(dataSet):
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

#在数据集中选出第axis个属性值为value的子集，并且去掉该属性值（删除已判断过的属性及其值）
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for examle in dataSet:
        if(examle[axis]==value):
            reducedDataSet=examle[:axis]
            reducedDataSet.extend(examle[axis+1:])
            retDataSet.append(reducedDataSet)
    return retDataSet

#在数据集中选出最优的划分属性，返回值为索引值
def chooseBestFeat(dataSet):
    numberFeats = len(dataSet[0])-2
    baseEntropy=calcShannonEnt(dataSet)
    baseInfoGain=0
    bestFeat=-1
    newEntropy=0
    for i in range(1, numberFeats+2):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        for one in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,one)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy-=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>baseInfoGain):
            bestFeat=i
            baseInfoGain=infoGain
    return bestFeat

def chooseBestFeat_infogainratio(dataSet):
    numberFeats=len(dataSet[0])-2
    baseEntropy=calcShannonEnt(dataSet)
    baseInfoGain=0
    bestInfoGainRatio = 0.0
    bestFeat=-1
    newEntropy=0
    for i in range(1, numberFeats+2):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy-=prob*calcShannonEnt(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain=baseEntropy-newEntropy
        if (splitInfo == 0):
            continue
        infoGainRatio = infoGain / splitInfo
        if(infoGainRatio > bestInfoGainRatio):
            bestFeat = i
            bestInfoGainRatio = infoGainRatio
    return bestFeat


# Create a terminal node value
def to_terminal(group):

    outcomes = [row[-1] for row in group]

    return max(set(outcomes), key=outcomes.count)


def majorityCnt(classList):

    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#对数据集创建树
def CreateID3Tree(dataSet,labels, maxDepth):
    classList=[example[-1] for example in dataSet]
    if(classList.count(classList[0])==len(classList)):
        return classList[0]
    if(len(dataSet[0])==1):
        return majorityClass(classList)
    bestFeat=chooseBestFeat_infogainratio(dataSet)
    bestLabel=labels[bestFeat]
    Tree={bestLabel:{}}
    del(labels[bestFeat])
    featVals=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featVals)
    for value in uniqueVals:
        subLabels=labels[:]
        #对分割后的数据子集递归创建树
        Tree[bestLabel][value]=CreateID3Tree(splitDataSet(dataSet,bestFeat,value),subLabels, maxDepth)
    return Tree

def CreateC45Tree(dataSet,labels, maxDepth):
    classList=[example[-1] for example in dataSet]
    if(classList.count(classList[0])==len(classList)):
        return classList[0]
    if(len(dataSet[0])==1):
        return majorityClass(classList)
    bestFeat=chooseBestFeat(dataSet)
    bestLabel=labels[bestFeat]
    Tree={bestLabel:{}}
    del(labels[bestFeat])
    featVals=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featVals)
    for value in uniqueVals:
        subLabels=labels[:]
        #对分割后的数据子集递归创建树
        Tree[bestLabel][value]=CreateC45Tree(splitDataSet(dataSet,bestFeat,value),subLabels, maxDepth)
    return Tree

def storeTree(trainTree, filename):

    fw = open(filename, 'wb')
    pickle.dump(trainTree, fw)
    fw.close()

def grabTree(filename):

    fr = open(filename, 'rb')
    return pickle.load(fr)
