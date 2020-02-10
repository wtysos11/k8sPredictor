# 固定读取部分
import os
fileName = 'result.txt'
dataPath = "E:\\code\\myPaper\\k8sPredictor"
#返回一个词典，词典的key为第二项，value为一个列表，所有列表都应该等长
def ReadDataFromFile():
    LocalPath = os.path.join(dataPath,fileName)
    reader = open(LocalPath,'r',encoding='utf-8')
    store = reader.readlines()
    reader.close()
    readKey = False
    lastkey = 'key'
    data = {}
    for line in store:
        if readKey == True:
            readKey = False
            lastkey = line[:-1]#删去行尾换行符
            data[lastkey] = []
        elif line[:-1] == 'keykey':
            readKey = True
        else:
            data[lastkey].append(int(line[:-1]))
    del store
    return data
        

def WriteDataToFile(storeData):
    LocalPath = os.path.join(dataPath,fileName)
    writer = open(LocalPath,'w',encoding='utf-8')
    for key in storeData:
        writer.write('keykey\n')
        writer.write(key+'\n')
        for ele in storeData[key]:
            writer.write(str(ele)+'\n')
    writer.close()

data = ReadDataFromFile()
from tslearn.utils import to_time_series_dataset
formatted_dataset = to_time_series_dataset(list(data.values()))

# 新方法开始部分

import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation,SymbolicAggregateApproximation
from tslearn.utils import to_time_series_dataset
import time
import numpy as np

# 想法二：自行计算趋势线中心和residual
# 1. 去除极端值（上下2%），对原有极端值地方使用线性插值或使用原有最大值（最大/最小抑制）
# 孤点不立：如果只有一个点，删除后去线性。如果有多个点，进行最大最小值抑制
# 2. 提取基线。提取方法可以使用滑动平均窗口。效果与PAA近似。可以直接使用PAA的结果
# 将原始数据与基线相减，得到residual。这个残余值应该是一个与一天内时间比较相关的量

# 计划：2月10、11日完成基线提取
# 2月11-12整合多层聚类算法
# 2月13日整合预测算法
# 2月14日整合流量调度系统
# 2月15日做实验

# 1. 对原始数据进行异常点去除
# 2. 进行与天数相同的点的PAA处理，提取基线数据
# 3. 使用k-means之类的算法进行一次简单聚类
# 4. 在简单聚类的基础上再进行一次复杂聚类
# 5. 基线进行预测，每个残余数据对震荡幅度进行预测


# 1.原始数据异常点去除实现：对原始数据进行排序。去除范围为前1%与后1%，合计2%左右
# 如果旁边没有点，则直接删除，然后对原有数据空缺部分进行线性插值。否则进行最大最小抑制

import numpy as np
ratio = 0.01#抑制前1%与后1%的值
stdData = formatted_dataset.copy()
for i in range(len(stdData)):
    ele = stdData[i].copy()
    ele = ele.reshape(ele.shape[0])
    ele.sort()#进行排序
    minNum,maxNum = ele[int(ratio*len(ele))],ele[-1*int(ratio*len(ele))]
    del ele #删除排序后的缓存
    # 进行异常点去除。
    # 如果这个点是一个孤立点，那么直接删除去线性。
    # 如果这个点不是孤立点，那么进行最大最小抑制
    variant = False #前一个点存在异常
    for index,ele in enumerate(stdData[i]):
        if ele > maxNum:
            if index+1 == len(stdData[i]): #是最后一个
                if variant and stdData[i][index-1] == maxNum:#上一个点是异常点，延续
                    stdData[i][index] = maxNum
                else:
                    stdData[i][index] = stdData[i][index-1]
            elif index == 0:#是第一个
                if stdData[i][index+1] < maxNum:# 下一个不是异常点，删除
                    stdData[i][index] = stdData[i][index+1]
                else:
                    stdData[i][index] = maxNum
            else:#不是最后一个也不是第一个
                if stdData[i][index-1]<maxNum and stdData[i][index+1]<maxNum: #孤立点，直接删除
                    stdData[i][index] = (stdData[i][index-1] + stdData[i][index+1])/2
                else:
                    stdData[i][index] = maxNum
        elif ele < minNum:
            if index+1 == len(stdData[i]): #是最后一个
                if variant and stdData[i][index-1] == minNum:
                    stdData[i][index] = minNum
                else:
                    stdData[i][index] = stdData[i][index-1]
            elif index == 0:#是第一个
                if stdData[i][index+1] > minNum:#下一个点不是异常点，删除
                    stdData[i][index] = stdData[i][index+1]
                else:
                    stdData[i][index] = minNum
            else:#不是最后一个也不是第一个
                if stdData[i][index-1]>minNum and stdData[i][index+1]>minNum: #孤立点，直接删除
                    stdData[i][index] = (stdData[i][index-1] + stdData[i][index+1])/2
                else:
                    stdData[i][index] = minNum
            variant = True
        else:# 不是异常点，进行标记
            variant = False

    

#归一化
from tslearn.preprocessing import TimeSeriesScalerMinMax
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
stdData = scaler.fit_transform(stdData)

# 2.对去除后的数据进行归一化处理

# 3.然后进行PAA处理，得到基线和残余值

# 4.对PAA后的数据进行简单k-means，聚类数量不超过10，分数按照CH分数判断，选出最大的

# 5.然后进行更进一步的聚类操作，对于每一个聚类进行详细聚类


# 产生的返回值为一个字典，字典内的每个键为n_samples*n_features格式
# 目标是拿到一次聚类所有相同聚类的数据并进行二次聚类
def getClstData(originData,clstData):
    ans = {}
    for index,ele in enumerate(clstData):
        if ele not in ans:
            ans[ele] = []
        ans[ele].append(index)
    return ans #可以通过numpy数组的索引，通过访问原始数据来访问聚类数据



# 6.对属于同一聚类的基线数据，进行预测。

# 7.对于聚类内部的每个元素，自行预测自己的残余值数据。（可以使用SAX进行噪点去除）

# 8.然后将数据整合起来，形成对最终结果的预测

# 9.进行实验