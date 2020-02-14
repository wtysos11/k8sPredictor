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

import numpy as np
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
originStdData = stdData # 保存，为了日后恢复
stdData = scaler.fit_transform(stdData)

# 3.然后进行PAA处理，得到基线和残余值
from tslearn.piecewise import PiecewiseAggregateApproximation
n_paa_segments = 20
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
paa_mid = paa.fit_transform(stdData)
paa_inv = paa.inverse_transform(paa_mid)
paa_inv = paa_inv.reshape(paa_inv.shape[0],paa_inv.shape[1])

# 4.对PAA后的数据进行简单k-means，聚类数量不超过10，分数按照CH分数判断，选出最大的
# 再进行rank-base处理，然后做简单聚类
from sklearn.cluster import MiniBatchKMeans,KMeans,DBSCAN,SpectralClustering,Birch
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score

n_cluster = 1000
s = time.time()
km = KMeans(n_clusters = n_cluster,random_state = 0)
y_pre = km.fit_predict(paa_inv)
e = time.time()
print(e-s,"s")
print(calinski_harabasz_score(paa_inv,y_pre))



# 5.然后进行更进一步的聚类操作，对于每一个聚类进行详细聚类

#----#
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


# 6.对属于同一聚类的基线数据，进行预测。

# 7.对于聚类内部的每个元素，自行预测自己的残余值数据。（可以使用SAX进行噪点去除）

# 8.然后将数据整合起来，形成对最终结果的预测

# 9.进行实验

# 对于聚类指标，可以不考虑没有聚类进来的时间序列，比如距离聚类中心一定距离的聚类的数量或比例等。依据实际情况来定。


# -------- #
# 接下来的考量是根据三种不同的方法来进行，以最终结果为准
#首先计算epsilon
data = paa_mid
epsilon = getEpsilon(data)
print('method1 episilon:',epsilon)
# 1. 朴素MiniBatchKmeans直接进行大规模聚类，聚类数量为1000
from sklearn.cluster import MiniBatchKMeans
kNum = 100
s = time.time()
km = MiniBatchKMeans(n_clusters = kNum,init_size = 3*kNum,random_state = 0,batch_size = 50)
y_pre = km.fit_predict(data)
e = time.time()
print('method1 :',e-s,' s')
playClus(data,y_pre,10)
total = 0
for i in range(kNum):
    total += getScore(data,y_pre,i,epsilon)
print(total/kNum)

# 两步聚类法分别使用rank-base和原始数据进行一次尝试
# 2. 使用Kmeans进行10以内的简单聚类。再用Kmeans对每个小聚类再进行聚类
from sklearn.cluster import KMeans
firstClus = 10
s = time.time()
km = KMeans(n_clusters = firstClus,random_state = 0)
y_pre = km.fit_predict(data)
#对每个聚类进行10聚类
secondClus = 10
totalScore = 0
# 对每个聚类产生一个新的数据，对这个数据进行再次聚类并且求出相关的系数
for k in range(firstClus):
    newData = data[y_pre == k]
    epsilon = getEpsilon(newData)
    km = KMeans(n_clusters = secondClus,random_state= 0)
    y_new_pre = km.fit_predict(newData)
    score = 0
    for i in range(secondClus):
        score += getScore(newData,y_new_pre,i,epsilon)
    totalScore += score/secondClus
print(totalScore/firstClus)
e = time.time()
print('method2:',e-s,'s')

# 3. 使用Kmeans进行10以内的简单聚类。再用其他聚类方式进行聚类

# 第一层聚类，要求：速度较快，容错较高
# 第一种情况：是否使用rank-base向量作为原始数据
# 第二种情况：是直接使用Birch进行聚类还是使用Birch聚类后再将kmeans导入
# 评价方式：三个：silhou系数，CH指数和我的分数

# 尝试一 普通数据，单纯Birch
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans,Birch
data = paa_mid

firstClus = 10
s = time.time()
y_pre = Birch(n_clusters=None,threshold = getEpsilon(data,0.8)).fit_predict(data)
e = time.time()
print(silhouette_score(data,y_pre))
print(getScore4Cluster(data,y_pre,0.4))
print(max(y_pre))
playClus(data,y_pre,10)

# 第一次聚类，4种情况，尽管按照通常指标是不同算法更优，但最终结果不一定。
# 1. 是否使用rank-base数据
# 给定一个m个特征的数据，返回其rank-base表示
def rankbased(origindata):
    ele = origindata.copy()
    ele.sort()
    ans = origindata.copy()
    for i in range(len(ans)):
        ans[i] = np.where(ele==ans[i])[0][0]
    del ele
    return ans

s = time.time()
first_clus = paa_mid.copy()
for i in range(len(first_clus)):
    first_clus[i] = rankbased(paa_mid[i])

e = time.time()
# 2. 方法：使用Birch，使用Birch后将数据导入到kMeans
## 只使用Birch
data = paa_mid
from sklearn.cluster import KMeans
firstClus = 10
s = time.time()
y_pre = Birch(n_clusters=None,threshold = getEpsilon(first_clus,0.8)).fit_predict(data)
e = time.time()
print(e-s,'s')
print(silhouette_score(data,y_pre))
print(getScore4Cluster(data,y_pre,0.4))
print('cluster Number',max(y_pre))
playClus(data,y_pre,10)

## 将Birch的值导入到Kmeans中
data = paa_mid
s = time.time()
y_pre = Birch(n_clusters=None,threshold = getEpsilon(first_clus,0.8)).fit_predict(data)
y_pre = KMeans(n_clusters = max(y_pre)+1,random_state = 0).fit_predict(first_clus)
e = time.time()
print(e-s,'s')
print(silhouette_score(data,y_pre))
print(getScore4Cluster(data,y_pre,0.4))
print('cluster Number',max(y_pre))
playClus(data,y_pre,10)

## 使用gap statistics判定k值
import pandas as pd
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters,5)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

# 第二次聚类，两种情况
# 第二次聚类可以选做，因为有些情况不一定需要第二次聚类。
# 1. 从效果上来说，gap statistics第一次聚类跑不出来效果。但是第二次聚类是可以的，因为第二次聚类的数量比较少。
# 从实际观察上来说，第二次聚类不应该超过10个类，因此第一次聚类的质量应该要足够的好。
# 尝试应该至少两次，即最大值和次大值
# --- # 测试用
data = paa_mid[y_pre==0]
s = time.time()
kNum = 5
km = KMeans(n_clusters = kNum,random_state = 0)
y_pre_t = km.fit_predict(data)
e = time.time()
print(e-s,'s')
print(silhouette_score(data,y_pre_t))
print('cluster Number',max(y_pre_t))
playClus(data,y_pre_t,5)

# 对于给定的一次聚类数据，自行进行二次聚类，并且返回聚类结果
# 第一种实现，使用gap statistic跑出k值
# 此处的data为特定的聚类结果，比如data[y_pre==k]
def getSecondClus_1(data):
    k, _ = optimalK(data, nrefs=5, maxClusters=8)
    km = KMeans(n_clusters = k,random_state= 0)
    y_pre = km.fit_predict(data)
    return y_pre

#第二种实现，使用中位数Birch聚类
def getSecondClus_2(data):
    epsilon = getEpsilonFromtiny(data)
    y_pre = Birch(n_clusters= None,threshold=epsilon).fit_predict(data)
    return y_pre

# 预测系统
# 普通ARIMA。因为数据比较少，普通ARIMA应该就够用了。
# 智能算法，比如人工神经网络。（暂时不考虑）

# 剩余预测：根据剩余值进行预测

# 返回值
# 要求：需要前面两次提取基线的时候保留平均值和方差以方便后面恢复
# 

# 最终评价指标：能够正确预测的流量数量（折返回去的预测值）

# 组装两次聚类
# 第一次聚类跑出来一个结果y_first_pre
# 对于第一次中的每一个聚类，进行再次聚类。同时可以考虑建立一个判别体系，对于已经足够好的聚类无需第二次聚类，判别方法以距离聚类中心的欧式距离为判定表尊
# 如果有90%的时间序列在范围内，可以考虑不用进行第二次聚类。否则则进行第二次聚类。这个范围应该由最终结果反推。
# 对每个第一次聚类跑出一个第二次聚类，平均十几个数据一个类别。
