#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/3/1 下午1:25
# @Author  : cicada@hole
# @File    : kNN.py
# @Desc    : k-近邻算法的python实现
# @Link    :


# 导入数据
from numpy import *
import operator # 运算符模块
import matplotlib.pyplot as plt
from os import listdir #列出目录的文件名

def createDataSet():
    group = array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ]) # 创建4*2矩阵
    labels = ['A', 'A', 'B', 'B'] # 对应标签
    return group, labels

# k-近邻算法
# （输入向量，输入样本集，标签，最近邻居的数目）
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # 4 shape为元祖（4，2）

    # 1. 计算当前点与已知点的距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet # 计算点之间的差 tile 将输入点 inX行重复4遍，列重复一遍
    sqDiffMat = diffMat**2  # 取平方
    sqDistance = sqDiffMat.sum(axis=1) # 1把该行相加，0把该列相加
    distances = sqDistance**0.5
    sortedDistIndicies = argsort(distances) #将距离升序排列
    # print(sortedDistIndicies)

    classCount={}
    # 2. 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    # 3. 排序
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

# 解析文本，返回矩阵和标签向量
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines) # 得到文件行数
    returnMat = zeros((numberOfLines,3)) # 文件行数*3 矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t') # 截取掉所有回车字符
        returnMat[index,:] = listFromLine[0:3] #前三个元素作为特征

        classLabelVector.append(int(listFromLine[-1])) # 列表元素为int
        index += 1
    return returnMat,classLabelVector

# 分析数据：使用matplotlib创建散点图
def createScatterPic(dataMat,label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,1], dataMat[:,2],
               15.0*array(label), 15.0*array(label))
    #展示第2、3列数据，第二参数：横轴纵轴长度 参数3：颜色范围序列
    plt.show()

# 归一化特征值
def autoNorm(dataSet): # 1000*3
    minVals = dataSet.min(0) # 从每列中选最小值 1*3
    maxVals = dataSet.max(0) # 从每列中选最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))# 1000*3
    m = dataSet.shape[0] #行数
    normDataSet = dataSet - tile(minVals, (m,1))
    # print(dataSet,'\n', maxVals,'\n',minVals,'\n', (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10 # 学习率
    dataDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(dataDataMat) #归一化
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  #用于测试的行数
    errorCount = 0.0
    for i in range(numTestVecs): # inX, 训练mat,训练label,k
        classifierResult = classify0(
            normMat[i,:],normMat[numTestVecs:m, :],
            datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):errorCount += 1.0
    print(" the total error rate is: %f"\
          % (errorCount/float(numTestVecs)))

def classifyPersion():
    resultList = ['不喜欢','喜欢一点','非常喜欢']
    percentTats = float(input("每周玩游戏时间？"))
    ffMiles = float(input("飞行里程？"))
    iceCream = float(input("每周吃冰淇淋的磅数？"))
    dataDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(dataDataMat)  # 归一化
    inArr = array([ffMiles, percentTats, iceCream])
    print('---inarr',inArr,'\n',minVals)

    classifierResult = classify0((inArr - minVals)/ranges,
                                 normMat, datingLabels, 3)
    print("你对这个人的喜欢程度：",resultList[classifierResult - 1])


#-----------手写识别系统-----------

# 格式化图像为向量
def img2vector(filename):
    returnVect = zeros((1,1024)) # 1x1024向量
    fr = open(filename)
    for i in range(32): #32x32的二进制图像
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
# 手写数字识别系统测试
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024)) #参数shape维度
    for i in range(m):
        fileNameStr = trainingFileList[i] # 0_0.txt
        # print(fileNameStr)
        fileStr = fileNameStr.split('.')[0]  #0_0
        classNumStr = int(fileStr.split('_')[0]) #对应的数字
        # print('-------classNum', classNumStr)
        hwLabels.append(classNumStr) #加入标签中
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s'
                                       %fileNameStr) # 把图像转换为向量
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,\
                                     trainingMat, hwLabels, 3)
        print("the classifier came back with: %d,\
                the real answer is : %d " \
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n total number of errors is: %d" %errorCount)
    print("\n total error tate is : %f" % (errorCount/float(mTest)))


def test():

    # group, labels = createDataSet()
    # print(group, labels)
    # print(classify0([0,0], group, labels, 3))
    # dataDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # # createScatterPic(dataDataMat, datingLabels)
    # autoNorm(dataDataMat)

    # datingClassTest()
    # classifyPersion()
    # vect = img2vector('digits/testDigits/0_13.txt') #32x32图像转换为1x1024向量
    # print(vect[0, 0:111])
    handwritingClassTest()


if __name__ == '__main__':

    test()



