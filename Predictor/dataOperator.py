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

from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation,SymbolicAggregateApproximation
from tslearn.utils import to_time_series_dataset
import time
import numpy as np
# 获取标准数据，并且修改原数据的值

class DataOperator:
    """
    这个类的目的是读取指定文件中的时间序列数据，并返回经过标准化的数据，然后对原始数据进行平均值和方差的调整使得其成为一个合适的测试数据
    最终数据：self.formatted_dataset 已经被修改过的测试用流量数据
    self.originStdData 用来逆标准化的流量数据
    self.stdData 已经标准化的数据
    """
    def __init__(self):
        data = ReadDataFromFile()
        formatted_dataset = to_time_series_dataset(list(data.values()))
        ratio = 0.05 #异常点数量
        #归一化
        scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
        stdData = scaler.fit_transform(formatted_dataset)

        # 为原有的均值设定为指定的方差和平均值（可能因为极端值的问题有些差别）
        # 将原始数据进行放缩，在原数据的基础上生成新的统一平均值数据
        DistAvg = 1000
        VarAvg = 2000
        self.formatted_dataset = formatted_dataset.copy()
        for i in range(len(formatted_dataset)):
            repres = stdData[i]
            self.formatted_dataset[i] = repres * np.sqrt(VarAvg) + DistAvg


        # 对于每一行的数据，得到一个绝对值排序结果，将最大的前5%排除出去。然后对空缺的位置进行线性差值
        for index,row in enumerate(stdData):
            element = abs(row)
            element = element.ravel()
            element.sort()
            maxNum = element[-1*int(ratio*len(element))]
            del element
            # 对于特殊情况，进行极值抑制
            # 只要能够找得到线性插值，就采用线性插值
            # 具体做法：不正常点（在最前或最后），直接采用最大值抑制
            # 使用一个数组进行一轮操作，将中间删除部分标出。
            # 对于被删掉的部分，寻找距离其最近的，两端进行线性插值。
            # previous为第一个异常点
            previous = -1
            for i,ele in enumerate(stdData[index]):
                if abs(ele) > maxNum:#这个点是异常点
                    if previous == -1:
                        previous = i
                else:
                    if previous != -1:
                        #开始进行从previous到i-1部分的线性插值
                        if previous == 0:# 如果previous是第一个，对全体进行最大值抑制
                            for vi in range(i):
                                if stdData[index][vi] < 0:
                                    stdData[index][vi] = -1*maxNum
                                else:
                                    stdData[index][vi] = maxNum
                        else:#不是，正常进行最大值抑制
                            dataRange = stdData[index][i] - stdData[index][previous-1]
                            number = i - (previous-1)
                            for vi in range(previous,i):
                                stdData[index][vi] = stdData[index][previous-1] + dataRange/number*(vi-previous+1) 
                        previous = -1

            #可能出现最后一个是异常点的情况，进行抑制
            if previous != -1:
                for vi in range(previous,len(stdData[index])):
                    if stdData[index][vi] < 0:
                        stdData[index][vi] = -1*maxNum
                    else:
                        stdData[index][vi] = maxNum


        # 2.对去除后的数据进行归一化处理
        #再进行一次归一化
        from tslearn.preprocessing import TimeSeriesScalerMinMax
        scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
        self.originStdData = stdData.copy()
        self.stdData = scaler.fit_transform(stdData)