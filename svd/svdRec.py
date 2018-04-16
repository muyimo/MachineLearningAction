#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/4/16 下午3:01
# @Author  : cicada@hole
# @File    : sedRec.py
# @Desc    : svd singular value decomposition
#           矩阵分解的一种类型
# @Link    :

'''
latent semantic indexing  LSI
latent semantic analysis  LSA
'''

from numpy  import *
import numpy as np
from numpy import linalg as la



def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]



# 相似度

def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))  # 范数 ,默认2


def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):

    num = (inA.T*inB).astype(float)  #python2这里的float()报错
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


'''
=======基于物品相似度的推荐引擎====
推荐未尝过的菜肴
1.寻找用户-物品矩阵中的0值
2.对这些物品评级打分(预计一个可能的评级)
3.评分从高到低排序,返回前n个
'''

# 给定相似度计算方法,计算用户对物品的估计评分
def standEst(dataMat, user, simMeas, item):

    n = shape(dataMat)[1]  #有多少物品吗
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        print('=========%d 第%d个物品========'%(item,j))
        userRating = dataMat[user, j]
        print('----用户打分为,',userRating)
        if userRating == 0 : continue
        overLap = nonzero(logical_and(dataMat[:, item].A>0,dataMat[:,j].A>0))[0]
        print('----overLap',overLap)
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap,item],
                                 dataMat[overLap,j])
            print(dataMat[overLap,item],'\n---',dataMat[overLap,j])
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

'''
======基于SVD的评分估计=======
'''
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    print('---U---',U,'\n')
    print('---Sigma---',Sigma, '\n')
    print('---VT--', VT, '\n')

    Sig4 = mat(eye(4)*Sigma[:4]) #对数据集进行svd分解,只利用90%能量的奇异值
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        # print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal



'''
=========具体原理=========
1.先针对指定用户,找到对应的未打分的物品xuhao
2.遍历这些序号
    3.依次去和其他物品的列表进行相似度比较
        比如第2个物品,和第一个作对比,首先取交集,之后算相似度
        之后相似度乘以该物品的打分,累加
4.最后返回预估得分
'''
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1] #find unrated items建立未评分的列表
    print('un------',unratedItems)
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems: #在所有未评分的物品上循环
        estimatedScore = estMethod(dataMat, user, simMeas, item) #产生评分
        itemScores.append((item, estimatedScore)) # 物品 评分列表
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]



'''
=====确定奇异值个数====
'''
from numpy import linalg as la
def compuNumOfSvd(data):
    U, Sigma, VT = linalg.svd(data)
    Sig2 = Sigma ** 2
    e1 = sum(Sig2)*0.9 #总能量的0.9
    e = sum (Sig2[:3]) # 前三个元素包含的能量
    if (e>e1):
        return 3




def test():
    Data = loadExData()
    # U, Sigma, VT = linalg.svd(Data)
    # print('----U', U)
    # print('----sigma', Sigma)
    # print('------vt', VT)
    #
    # # 重构原始矩阵
    # Sig3 = matrix([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])  # 原始
    # new = U[:, :3] * Sig3 * VT[:3, :]
    # print(new)


    myMat = matrix(Data)
    # print(myMat[:,0],'/n',myMat[:,3])
    # sim1 = ecludSim(myMat[:,0],myMat[:,3])
    # sim2 = pearsSim(myMat[:,0],myMat[:,3])
    # sim3 = cosSim(myMat[:,0],myMat[:,3])
    # print(sim1,sim2,sim3)

    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
    myMat[3, 3]=2
    print(myMat)
    rc1 = recommend(myMat,2)
    rc2 = recommend(myMat, 2,estMethod=svdEst)
    print(rc1,'\n----\n',rc2)







if __name__ == '__main__':
    test()
