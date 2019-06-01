# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:22:48 2018

@author: Administrator
"""
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import csv
import sys



'''
加载数据集,获得训练数据和测试数据 2500  920  8：2
'''
def loadDataSet(filename1,filename2,num):
    '''
    加载HMM总数据
    '''
    f11 = open(filename1,'rb')
    data1 = pickle.load(f11)
    f11.close()
    f12 = open(filename2,'rb')
    data2 = pickle.load(f12)
    f12.close()
    '''
    获得统计信息
    '''
    filePath1 = "./data/15023.csv"
    filePath2 = "./data/18017.csv"
    dataSet1 = processData(filePath1)
    dataSet2 = processData(filePath2)
    firOrder1, secOrder1, thiOrder1 = getSignal(dataSet1)
    firOrder2, secOrder2, thiOrder2 = getSignal(dataSet2)
    '''
    2736个训练数据
    '''
    m1 = 2000
    n1 = 736
    le1 = 2736
    trainingData = np.zeros((le1,num + 3))
    trainingData[0:m1,0:num] = data1[0:m1,:]
    trainingData[m1:le1,0:num] = data2[0:n1,:]#使用全部特征
    
    trainingData[0:m1,num] = np.array(firOrder1)[0:m1]
    trainingData[m1:le1,num] = np.array(firOrder2)[0:n1]
    
    trainingData[0:m1,num+1] = np.array(secOrder1)[0:m1]
    trainingData[m1:le1,num+1] = np.array(secOrder2)[0:n1]
    
    trainingData[0:m1,num+2] = np.array(thiOrder1)[0:m1]
    trainingData[m1:le1,num+2] = np.array(thiOrder2)[0:n1]
    
    labelMat = np.ones((le1,1))#列向量
    labelMat[m1:le1,:] = -1 * np.ones((n1,1))
    '''
    1026个测试数据
    '''
    m2 = 500
    n2 = 184
    le2 = 684
    
    testData = np.zeros((le2,num+3))
    testData[0:m2,0:num] = data1[m1:m1+m2,:]
    testData[m2:le2,0:num] = data2[n1:n1+n2,:]#使用全部特征  
    
    testData[0:m2,num] = np.array(firOrder1)[m1:m1+m2]
    testData[m2:le2,num] = np.array(firOrder2)[n1:n1+n2]
    
    testData[0:m2,num+1] = np.array(secOrder1)[m1:m1+m2]
    testData[m2:le2,num+1] = np.array(secOrder2)[n1:n1+n2]
    
    testData[0:m2,num+2] = np.array(thiOrder1)[m1:m1+m2]
    testData[m2:le2,num+2] = np.array(thiOrder2)[n1:n1+n2]
    
    labelMatTest = np.ones((le2,1))#列向量
    labelMatTest[m2:le2,:] = -1 * np.ones((n2,1))
    
    return trainingData,labelMat.T,testData,labelMatTest.T
	
def processData(filename):
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        # decrease the maxInt value by factor 10  
        # as long as the OverflowError occurs.  
        
        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt / 10)  
            decrement = True 
    
    
    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile)
        
        '''
        获得表头信息
        '''
        for row in reader:
            header = row
            break
        
        '''
        获得每个timeseries,保存在dataSet中
        '''
        m = len(header)
        dataSet = []#timeseries
        for row in reader:                
            if row[2] == 'normal' and float(row[4]) > 22 and float(row[4]) < 24:
                dataSet.append(row[m-1])
        csvfile.close()
        return dataSet
    
'''获取每个阻断事件的阻断流部分,并获得阻断事件的统计信息'''
def getSignal(dataSet):
    sig = []#原始的每个信号
    for item in dataSet:
        tlist = []#单个事件信息
        for i in item.strip('[], ').split(','):#去除首‘[’和尾部‘]’
            tlist.append(float(i))
        sig.append(tlist)
    k = [1, 2, 3]
    firOrder = []
    secOrder = []
    thiOrder = []
    for i in range(len(sig)):
        x = sig[i]
        n = len(x)
        mean = sum(np.power(x, k[0]))/n
        firOrder.append(mean)
        var = sum(np.power(x-mean, k[1]))/n
        secOrder.append(var)
        thiOrder.append((sum(np.power(x-mean, k[2]))/n)/np.power(var, 3))
    return firOrder, secOrder, thiOrder        

'''
数据归一化
'''
def normalizedData(data):

    minVals = data.min(0)
    maxVals = data.max(0)#获得每列的最小值和最大值
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = np.shape(data)[0]
    normData = data - np.tile(minVals,(m,1))
    normData = normData/np.tile(ranges,(m,1))
    return normData
    

    
    

'''
加载HMM数据集,保存到文件,方便使用
只使用15023和18017两次实验的数据
'''
filename1 = "./data/hmm_15023.pkl"
filename2 = "./data/hmm_18017.pkl"
num = 8#总特征数
trainingData,labelMat,testData,labelMatTest = loadDataSet(filename1,filename2,num)#获得数据
#trainingData = normalizedData(trainingData)
#testData = normalizedData(testData)
outputPath = "./data/trainingData.pkl"
output = open(outputPath, 'wb')
pickle.dump(trainingData,output)
output.close()

outputPath1 = "./data/labelMat.pkl"
output1 = open(outputPath1, 'wb')
pickle.dump(labelMat,output1)
output1.close()

outputPath2 = "./data/testData.pkl"
output2 = open(outputPath2, 'wb')
pickle.dump(testData,output2)
output2.close()

outputPath3 = "./data/labelMatTest.pkl"
output3 = open(outputPath3, 'wb')
pickle.dump(labelMatTest,output3)
output3.close()