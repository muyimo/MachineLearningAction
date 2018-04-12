#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/2/24 上午11:20
# @Author  : fwj
# @Site    : 
# @File    : old-boost.py
# @Software: PyCharm

from numpy import ones, shape, mat, zeros, inf, matrix,array

def loadSimpData():
    datMat=matrix([[1.,2.1],
        [2.,1.1],
        [1.3,1.],
        [1.,1.],
        [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

#单层决策树生成函数
#通过阈值比较分类，阈值一边的数据分到类别+1，另一边分到-1
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):

    # print("------dimen",dimen)
    retArray = ones((shape(dataMatrix)[0], 1)) #shape维数,[0]取行数，生成 1列的1矩阵
    # print(retArray)
    if threshIneq == 'lt':
        # print("---------dataMatrix[:, dimen]")
        # print(dataMatrix[:, dimen])
        # print("---------threshVal")
        # print( retArray[dataMatrix[:, dimen] <= threshVal ])
        retArray[dataMatrix[:, dimen] <= threshVal ] = -1.0 #意思是,值小于阈值的,把他设置为-1
        # print("-----------retArray",retArray)
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

'''
=====用于找单层最佳决策树=====
原理:
    1.遍历特征
    2.  针对单个特征,取最小,最大,求区间,设置步长,由小到大逐步更新
    3.      针对单个步长,设置lt,gt,去算出划分的结果
    4.划分的结果跟标签集对比,求误差,更新相关参数
    
'''
def buildStump(dataArr, classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    # numSteps用于在特征的所有可能值上进行遍历
    # bestStump用于存储给定权重向量D时所得到的最佳单层决策树
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)));
    minError = inf #初始化为正的无穷大，用于之后寻找可能的最小错误率
    #第一层循环 在所有特征上遍历
    for i in range(n):
        print("======================遍历第",i,"个特征==============")
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps #步长
        # print(stepSize)

        #第二层循环 在步长范围上遍历
        for j in range(-1, int(numSteps)+1):
            # print("================遍历第", j, "个步长==============")
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize) #特征值从最小 变到最大
                # print("=============此刻阈值为:",rangeMin, ",",threshVal)

                # 根据阈值划分一个结果,然后跟label对比,选择最好的
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s,the wegihted error is %.3f" %\
                      (i, threshVal, inequal, weightedError)
                     )
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                print("                                            ")
            print("                                            ")
    print("                                            ")
    print("                                            ")
    print("                                            ")
    # print("===========best classESt",bestClasEst)
    return bestStump, minError, bestClasEst

def test():
    matrix = [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ]
    # print(shape(matrix),shape(matrix)[0])
    # print(ones((4,1))) #python的二维数组要用两个括号括起来
    tp = (1, 2, 3, 4, 5,)
    dataArr = (1, 2, 2, 2, 2, 2, 2, 3)
    print(mat(dataArr))
    # print(tp[:-1],shape(tp))
    # print(shape(matrix[:,2]))
    # print(inf)#inf最大下界

def main():
    print("=============== main start ===============")
    D = mat(ones((5, 1))/5)
    # print(D)
    datMat, classLabels = loadSimpData()
    print("-----datamat",datMat)
    print("-----classLabels",classLabels)

    bestStump, minError, bestClasEst = buildStump(datMat,classLabels,D)
    print(bestStump, minError, bestClasEst)

if __name__ == "__main__":
    main()
    l = [
        [1,2,3,4],
        [5, 6, 7, 8]
        ]
    l = array(l)
    # print(l[:,1]) #所有行的第一列数据
    # print(l[1,:]) #第一行的所有列数据

