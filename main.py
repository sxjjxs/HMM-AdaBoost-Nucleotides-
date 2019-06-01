# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:18:19 2018

@author: Administrator
"""
import sys
import pickle 
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics 
from sklearn import svm
from sklearn import metrics
import time
import matplotlib.pyplot as plt
start_time1 = time.clock()
'''
加载数据集
'''

filename1 = "./data/trainingData.pkl"
f1 = open(filename1,'rb')
trainingData = pickle.load(f1)
f1.close()

filename2 = "./data/labelMat.pkl"
f2 = open(filename2,'rb')
labelMat = pickle.load(f2)
f2.close()

filename3 = "./data/testData.pkl"
f3 = open(filename3, 'rb')
testData = pickle.load(f3)
f3.close()

filename4 = "./data/labelMatTest.pkl"
f4 = open(filename4, 'rb')
labelMatTest = pickle.load(f4)
f4.close()

'''Adaboost'''

adt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         learning_rate=1.5,
                         n_estimators=70)
adt.fit(trainingData, labelMat.T)
end_time1 = time.clock()
time1 = end_time1-start_time1
print('\nRunning time1:', time1)
print('+++++++++++_______+++++++++++')
print(adt.score(trainingData, labelMat.T))
print(adt.score(testData, labelMatTest.T))
print(metrics.f1_score(adt.predict(testData),labelMatTest.T))
y_score = adt.fit(trainingData, labelMat.T).decision_function(testData)
print(metrics.roc_auc_score(labelMatTest.T, y_score))
pre_adt = adt.predict(testData)
t = labelMatTest.T
print(metrics.confusion_matrix(labelMatTest.T, adt.predict(testData)))

count_adt_pos = 0
count_adt_neg = 0
x = []
for i in range(len(pre_adt)):
    x.append(i)
    if pre_adt[i] == -1:# and pre_adt[i] != t[i]:
        count_adt_pos +=1
    if pre_adt[i] == 1: #and pre_adt[i] != t[i]:
        count_adt_neg +=1

print('+++++++++++_______+++++++++++')

"""
trainScore = []
testScore = []
f1Score = []
times = []
numIt = []
estimator_errors = []
auc = []
count = []
for i in range (1,71):
    if i == 1 or i % 5 == 0:
        adt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         learning_rate=1.5,
                         n_estimators=i)
        adt.fit(trainingData, labelMat.T)
        '''得到分类错误样本的个数'''
        pre = adt.predict(testData)
        c = 0
        for j in range(len(pre)):
            if pre[j] != labelMatTest[:,j]:
                c += 1
        count.append(c)
        '''训练错误率'''
        trainScore.append(adt.score(trainingData, labelMat.T))
        '''测试错误率'''
        testScore.append(adt.score(testData, labelMatTest.T))
        '''f1-score'''
        f1Score.append(metrics.f1_score(adt.predict(testData),labelMatTest.T))
        '''AUC值'''
        y_score = adt.fit(trainingData, labelMat.T).decision_function(testData)
        auc.append(metrics.roc_auc_score(labelMatTest.T, y_score))
        '''所用时间'''
        end_time1 = time.clock()
        time1 = end_time1-start_time1
        times.append(time1)
        numIt.append(i)

'''画图'''
fig = plt.figure(figsize=(9,5))
fig.clf()
plt.plot(numIt,trainScore, linewidth=3,label='training accuracy')
plt.plot(numIt,testScore, linewidth=3,label='test accuracy')
#plt.plot(numIt,f1Score, linewidth=3,label='f1_score')
plt.legend(loc='left right',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.ylim(-0.02,0.25)
plt.xlabel('n-estimator', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.show()

"""