#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/3/5 下午1:21
# @Author  : cicada@hole
# @File    : trees.py
# @Desc    : 第二章 决策树
# @Link    :

from math import log

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
        prob = float(labelCounts[key]) / numEntries
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

def test():
    myDat, labels = createDataSet()
    print(myDat)
    # myDat[0][-1] = 'maybe'
    shannonEnt = calcShannonEnt(myDat) #熵越高，混合的数据越多
    print(shannonEnt)

if __name__ == '__main__':
    test()