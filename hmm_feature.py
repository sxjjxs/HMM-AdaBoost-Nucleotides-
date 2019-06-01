# -*- coding: utf-8 -*-
"""
得到每个阻断事件对应的HMM模型参数信息
"""

import numpy as np
from hmmlearn import hmm
import csv
import sys
import pickle


'''
获得每个阻断事件的HMM模型参数(转移矩阵、均值、方差)
'''
def getData(filePath,outputPath):

    num = 8#含有特征数量
    allDataSet = processData(filePath)
    allTrainingList = []#列表形式的训练集
    for item in allDataSet:
        trainingList = []#单个事件信息
        for i in item.strip('[], ').split(','):#去除首‘[’和尾部‘]’
            trainingList.append(float(i))
        '''
        获得HMM模型
        '''
        trainingSet = np.array(trainingList).reshape(-1,1)
        model = hmm.GaussianHMM(n_components = 2,n_iter = 1000)#创建对象
        model.fit(trainingSet)#训练
    
        data = toVec(model.transmat_,model.means_,model.covars_).reshape(1,-1)#将模型数据转换成行向量
        
        allTrainingList.append(data)#存储列表中
    
    m = len(allTrainingList)
    allTrainingData = np.zeros((m,num))
    for p in range(m):
            allTrainingData[p] = allTrainingList[p]
    
    output = open(outputPath, 'wb')
    pickle.dump(allTrainingData,output)
    output.close()

'''
解决 Error: field larger than field limit (131072)
'''
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

'''
把transmat,startprob,means,covars转成行向量
'''
def toVec(transmat,means,covars):    

    dataList = []
    for m in range(2):
        for n in range(2):
            dataList.append(transmat[m][n])#完成transmat的转换
    for i in range(2):
#        dataList.append(startprob[i])
        dataList.append(means[i][0])#完成means的转换
        dataList.append(covars[i][0][0])#完成covars的转换
    data = np.array(dataList)#列表转向量
    return data

"""
15012和15023为AA3
18003和18017为GA3
"""
filePath1 = "./data/15012.csv"
outputPath1 = "./data/hmm_15012.pkl"
getData(filePath1,outputPath1)

filePath2 = "./data/18003.csv"
outputPath2 = "./data/hmm_18003.pkl"
getData(filePath2,outputPath2)

filePath3 = "./data/15023.csv"
outputPath3 = "./data/hmm_15023.pkl"
hmm_modle.getData(filePath3,outputPath3)

filePath4 = "./data/18017.csv"
outputPath4 = "./data/hmm_18017_AU.pkl"
hmm_modle.getData(filePath4,outputPath4)

