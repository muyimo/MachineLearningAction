#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/4/11 下午5:12
# @Author  : cicada@hole
# @File    : adaboost.py
# @Desc    : 
# @Link    :

from numpy import matrix, ones, shape,zeros,log,multiply,exp,sign
from ml.adaboost import boost



'''
============基于单层决策树的AdaBoost训练=========
D: 数据点权重向量,adaboost会加重错分的数据点的权重
1. 查找最佳单层决策树
2. 将其加入决策数组
3. 计算alpha
4. 计算样本权重向量D
5. 更新累计类别估计值
6. 如果错误率为0就退出

'''
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = matrix(ones((m,1))/m)
    aggClassEst = matrix(zeros((m,1)))

    # 迭代次数numIt
    for i in range(numIt):
        print("--------------迭代第 ", i," 次----------")
        #classEst 训练器的分类结果
        bestStump, error, classEst = boost.buildStump(dataArr,classLabels,D)
        print("每个数据点权重向量 D:",D.T)

        # error = 错分样本数/总数
        # alPha= 1/2ln((1-e)/e) 每个分类器的权重值
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ",classEst.T)

        expon = multiply(-1*alpha*matrix(classLabels).T,classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst  # 分类器累加
        print("aggClassEst: ",aggClassEst.T)

        aggErrors = multiply(sign(aggClassEst) != matrix(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate,"\n")

        if errorRate == 0.0 :
            print("----------------------迭代Ok,退出-----------------")
            break
    return weakClassArr


'''
======AdaBoost分类函数=====
利用训练出来的多个弱分类器进行分类

'''
def adaClassify(datToClass, classifierArr):
    dataMatrix = matrix(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = matrix(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = boost.stumpClassify(dataMatrix,classifierArr[i]['dim'],
                                       classifierArr[i]['thresh'],
                                       classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)




if __name__ == '__main__':
    dataMat, classLabels = boost.loadSimpData()
    classifierArray = adaBoostTrainDS(dataMat,classLabels,9)
    rt = adaClassify([0,0],classifierArray)
    print("---预测的标签:",rt)











