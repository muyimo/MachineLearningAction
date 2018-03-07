#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/3/5 下午1:21
# @Author  : cicada@hole
# @File    : trees.py
# @Desc    : 第二章 决策树
# @Link    :

from math import log
import random
import json
import operator

'''
计算给定数据集的香农熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 实例总数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries # 特征i所占的比例
        shannonEnt -= prob * log(prob,2)
    return shannonEnt


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']]
    labels = ['no surfacing', 'flippers'] #浮出水面是否能生存 是否有脚蹼
    return dataSet, labels

'''
    划分数据集
    1.如果第i个特征值为value
    2.剩余的vec从start-i-1，i+1~end，组装成新的vec
    3.返回挑选后的数据集合
'''
def splitDataSet(dataSet, axis, value):# 待分数据集、第几个特征特征、特征返回值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1 :]) # 为了集合中筛出指定的特征featVec[:axis]
            # print("------reducedFeatVec", reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet



'''
选择最好的数据集划分方式：
    1.获取数据集的特征数，计算原始香农熵
    2.遍历特征，根据单个特征值，划分数据集，求得原始熵-划分集熵，即信息增益
    3.更新信息增益，增益最大的特征即为最好的特征
    备注：熵越大，信息量越大，信息中如果全为同样的值，则熵为0，信息量最小
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 # 特征数
    baseEntropy = calcShannonEnt(dataSet) #计算香农熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures): # 遍历特征
        featList = [example[i] for example in dataSet] #第i个特征的所有值
        uniqueVals = set(featList) # 单个特征中的无序不重复元素
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i ,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy #唯一特征得到的熵
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature # 返回第i个特征

'''
递归构建决策树：
    1.递归结束条件：程序遍历完数据集，或者每个分支下所有实例都具有相同分类
    2.多数表决法
    3.返回出现次数最多的分类
'''
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter,
                              reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时，返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    # 得到列表包含的所有属性值
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(\
            dataSet, bestFeat, value), subLabels)
    return myTree


def test():
    myDat, labels = createDataSet()
    print(myDat)
    # myDat[0][-1] = 'maybe'
    # shannonEnt = calcShannonEnt(myDat) #熵越高，混合的数据越多
    # print(shannonEnt)
    retDataSet = splitDataSet(myDat, 1 ,0) #对数据集mgDat，第1个特征为0进行划分
    print(retDataSet)
    bestFeature = chooseBestFeatureToSplit(myDat) #第i个特征
    print(bestFeature)

def test1():
    pass


if __name__ == '__main__':
    test()