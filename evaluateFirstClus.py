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

# 时间序列：异常点排除一为方法1
# 时间序列：异常点排除方法2：对原数据进行归一化。将所有数据减去平均值的绝对值后排序，去除5%，并进行线性插值

ratio = 0.05 #异常点数量

#归一化
from tslearn.preprocessing import TimeSeriesScalerMinMax
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
stdData = scaler.fit_transform(formatted_dataset)

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
originStdData = stdData.copy()
stdData = scaler.fit_transform(stdData)
# 新方法开始部分

from sklearn.cluster import MiniBatchKMeans,KMeans,DBSCAN,SpectralClustering,Birch
from sklearn.metrics import silhouette_score,calinski_harabasz_score

from statsmodels.tsa.arima_model import ARIMA,ARMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# 数据处理部分

#工具人函数
# 求数据距离中心的平均距离
def getAvgDist(data):
    center = sum(data)/data.shape[0]
    score = 0
    for i in range(len(data)):
        t = getDist(center,data[i])
        score += t
    return score/len(data)

# 新的评价指标：只考虑在基线中心附近一定范围内的时间序列的内聚度，超过这个范围则完全不考虑。
# getCenter函数，目标是得到指定聚类的中心
def getCenter(originData,y_pre,clusNum):
    clusData = originData[y_pre==clusNum]
    if clusData.shape[0] == 0:#如果聚类为空，返回0
        return np.zeros(originData.shape[1])
    else:
        center = sum(clusData)/clusData.shape[0]
        return center

# 衡量两个给定时间序列之间的欧式距离（默认其等长）
def getDist(vec1,vec2):
    vec1 = vec1.ravel()
    vec2 = vec2.ravel()
    return np.linalg.norm(vec1 - vec2)

# 给定一个聚类以及一个范围，得到这个聚类的分数。分数越小聚类效果越好
# episilon是一个范围，如果聚类中心到聚类的欧式距离超过了这个范围，则完全不考虑
def getScore(originData,y_pre,clusNum,epsilon):
    center = getCenter(originData,y_pre,clusNum)
    ans = 0
    num = 0
    clusNumber = 0
    for index,clus in enumerate(y_pre):
        if clus == clusNum:
            clusNumber += 1
            dist = getDist(center,originData[index])
            if dist < epsilon:
                ans += dist
                num += 1
    if num == 0:
        return 100*epsilon
    else:
        return ans/num/(num/clusNumber)

# episilon的计算可以由预先进行，先对原始数据进行一个100-means分类，以前10个聚类中50%聚类与中心的距离的平均数计算出来
# ratio是比例，默认取最小的50%
def getEpsilon(originData,ratio = 0.5):
    km = KMeans(n_clusters = 100,random_state = 0)
    y_pre = km.fit_predict(originData)
    score = 0
    num = 0
    for k in range(100):
        center = getCenter(originData,y_pre,k)
        l = []
        for index in np.where(y_pre == k)[0]:
            l.append(getDist(center,originData[index]))

        l.sort()
        if int(len(l)*ratio)!=0:
            score += l[int(len(l)*ratio)]
            num += 1
    return score / num

def getEpsilonFromtiny(originData):
    center = sum(originData)/originData.shape[0]
    l = []
    for index in range(len(originData)):
        l.append(getDist(center,originData[index]))
    
    l.sort()
    return l[int(len(l)*0.5)]

#简易的可视化函数，能够可视化聚类结果
# first_clus为原始聚类数据，y_pre为聚类结果，num为观看的聚类的数量。
def playClus(first_clus,y_pre,num):
    for k in range(num):
        count = 0
        for i in range(len(y_pre)):
            if y_pre[i] == k:
                plt.plot(first_clus[i])
                count += 1
        print(count)
        plt.show()

# ratio是getEpsilon取的范围阈值
def getScore4Cluster(originData,y_pre,ratio):
    epsilon = getEpsilon(originData,ratio)
    clusNum = max(y_pre)
    score = 0
    for i in range(clusNum+1):
        score += getScore(originData,y_pre,i,epsilon)
    return score/(clusNum+1) 

# 给定一个m个特征的数据，返回其rank-base表示
def rankbased(origindata):
    ele = origindata.copy()
    ele.sort()
    ans = origindata.copy()
    for i in range(len(ans)):
        ans[i] = np.where(ele==ans[i])[0][0]
    del ele
    return ans
#################################################################
# 初始，根据原始数据计算新数据
# 从paa这里

# 需要在聚类前将训练数据划分完毕

s = time.time()
ratio = 0.9
n_paa_segments = 18
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
paa_mid = paa.fit_transform(stdData[:,:int(ratio*stdData.shape[1])])
paa_mid = paa_mid.reshape(paa_mid.shape[0],paa_mid.shape[1])

first_clus = paa_mid.copy()
for i in range(len(first_clus)):
    first_clus[i] = rankbased(paa_mid[i])

#################################################################
# 第一次聚类使用Birch跑出初始，然后使用Kmeans细分。数据使用rank-base
# 改进：直接使用原始数据，调整Birch的threshold

data = first_clus

y_pre = Birch(n_clusters=None,threshold = getEpsilon(data,0.8)).fit_predict(data)
y_pre = KMeans(n_clusters = max(y_pre)+1,random_state = 0).fit_predict(data)

e = time.time()
print(e-s,'s')

#聚类分布情况
print('聚类分布情况')
clusDistribution = []
for k in range(int(max(y_pre)+1)):
    clusDistribution.append(len(stdData[y_pre==k]))
clusDistribution.sort()
plt.plot(clusDistribution)
plt.show()
from scipy.stats import spearmanr

def evaluateResult(clusResult,bigClusThre = 5):
    print('聚类数量：',max(clusResult)+1)
    t = []
    for i in range(int(max(clusResult))):
        t.append(len(stdData[i==clusResult]))
    t = np.array(t)
    print('平均聚类数',np.average(t))
          
    bigClus = 0#大聚类个数
    tsInBigClus = 0#时间序列中的大时间聚类
    BgScore = 0
    BgCoeff = 0
    for k in range(int(max(clusResult))+1):# 枚举所有的聚类
        # 计算聚类中心
        l = [] #相关系数排名
        s = [] #
        sigma = stdData[clusResult==k]
        if len(sigma)< bigClusThre:
            continue
        center = sum(sigma)/sigma.shape[0]
        bigClus += 1#聚类元素大于5，视为大聚类
        
        for index in np.where(clusResult==k)[0]:
            tsInBigClus += 1
            l.append(spearmanr(center,stdData[index])[0])
            s.append(getDist(center,stdData[index]))
    
        l.sort()
        s.sort()
        spscore = 0 # 分位点数据
        if len(l) == 1:
            if math.isnan(l[0]):
                spscore = -1
            else:
                spscore = l[0]
        else:
            if math.isnan(l[int(len(l)*0.8)]):
                spscore = -1
            else:
                spscore = l[int(len(l)*0.8)]   
        BgCoeff +=spscore
        BgScore += s[int(len(s)*0.8)] 
    
    BgCoeff/=bigClus
    BgScore/=bigClus
    print('大聚类的时间序列数占比',tsInBigClus/len(stdData))
    print('大聚类中8分位点的平均距离',BgScore)
    print('大聚类中8分位点的相关系数',BgCoeff)

evaluateResult(y_pre,40)

#给出响应时间列表与SLA的值，返回超过SLA的平均时间
def getResVal(res,sla):
    count = 0
    t = 0
    for i in range(len(res)):
        if res[i]>sla:
            count += 1
            t += res[i]-sla
    if count != 0:
        return t/count
    else:
        return 0