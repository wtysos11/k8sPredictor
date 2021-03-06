# Birch论文 A BIRCH-Based Clustering Method for Large Time Series Databases
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

# 为原有的均值设定为指定的方差和平均值（可能因为极端值的问题有些差别）
# 将原始数据进行放缩，在原数据的基础上生成新的统一平均值数据
DistAvg = 2000
VarAvg = 15000
for i in range(len(formatted_dataset)):
    repres = stdData[i]
    formatted_dataset[i] = repres * np.sqrt(VarAvg) + DistAvg


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
from sklearn.metrics import silhouette_score,calinski_harabasz_score,mean_squared_error

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

# 需要在聚类前将训练数据划分完毕。ratio必须要精心选择使得paa的值能够成为整数

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
s = time.time()
y_pre = Birch(n_clusters=None,threshold = getEpsilon(data,0.8)).fit_predict(data)
y_pre = KMeans(n_clusters = max(y_pre)+1,random_state = 0).fit_predict(data)
e = time.time()

#################################################################
# 第二次聚类使用10以内间隔2的gap statistics。聚类对象为残差
# 改进：可以考虑聚类对象是残差或直接是标准数据
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
    for gap_index, k in enumerate(range(1, maxClusters,2)):
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

# 对于给定的一次聚类数据，自行进行二次聚类，并且返回聚类结果
# 第一种实现，使用gap statistic跑出k值
# 此处的data为特定的聚类结果，比如data[y_pre==k]
def getSecondClus_1(data):
    k, _ = optimalK(data, nrefs=5, maxClusters=10)
    km = KMeans(n_clusters = k,random_state= 0)
    y_pre = km.fit_predict(data)
    return y_pre

#第二种实现，使用中位数Birch聚类
def getSecondClus_2(data):
    epsilon = getEpsilonFromtiny(data)
    y_pre = Birch(n_clusters= None,threshold=epsilon).fit_predict(data)
    return y_pre

from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr
import math
from sklearn.metrics import pairwise_distances

#第三种实现，使用Agglomerative非欧距离聚类
#速度上来说是普通MSE的4倍
def getTrendScore(x,y):
    v,p = spearmanr(x.ravel(),y.ravel())
    # 如果v<=0，或者p>=0.05，说明两者无线性关系，距离趋于近无穷大
    if math.isnan(v):
        return pow(math.e,10)
    elif v<=0.1 or p>=0.05:
        return pow(math.e,10)
    else:
        return pow(math.e,1/v)

def trend_affinity(X):
    return pairwise_distances(X, metric=getTrendScore,n_jobs=-1)
def getSecondClus_3(data):
    distance_matrix = trend_affinity(data)
    model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',distance_threshold = pow(math.e,1/0.5))
    y_pre = model.fit_predict(distance_matrix)
    return y_pre

#对于标准数据耗时300s左右
cluster_num = max(y_pre)
totalClusterNum = 0 #前面最小的聚类数
secondClusans = np.zeros(len(y_pre))# 全0的数组，用来保存最后的结果
for k in range(cluster_num + 1):
    paaData = paa_mid[y_pre == k]
    originData = stdData[:,:int(ratio*stdData.shape[1])][y_pre==k]
    originData = originData.reshape(originData.shape[0],originData.shape[1])
    second_iter = np.where(y_pre == k)[0]

    if len(originData) < 15:#如果聚类过小，不进行第二次聚类
        for index,ele in enumerate(second_iter):
            secondClusans[ele] = totalClusterNum
        totalClusterNum += 1
    else:
        second_y = getSecondClus_1(originData)
        for index,ele in enumerate(second_iter):
            secondClusans[ele] = second_y[index] + totalClusterNum
        totalClusterNum += max(second_y) + 1 

# 第二次聚类完毕，得到的结果是secondClusans，里面是按顺序存储的聚类数据。totalClusterNum是总聚类数量
# 初步得到的聚类数量为1463个聚类
#################################################################
# 第三步：整体预测。提取出聚类中心并对其进行预测，将结果作为聚类中所有值的最终结果
store = []
for k in range(totalClusterNum):
    stdClusData = stdData[secondClusans == k]
    store.append(sum(stdClusData)/stdClusData.shape[0])


#################################################################
# 第四步：预测。建立模型对该数据进行预测（决策树回归模型），接受480个数据预测20个
# 改进：可以考虑与基线+残差双ARIMA预测方法进行比较，以及要与基础的普通决策树回归预测方法进行比较
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
#返回滑动窗口
def getWindow(data,window_size):
    x = []
    for t in range(len(data)-window_size+1):
        a = data[t:t+window_size]
        x.append(a)
    x = np.array(x)
    x = np.reshape(x,(len(x),window_size))
    return x

# 用于GridSearchCV调参的
def print_best_score(gsearch,param_test):
     # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


from sklearn.preprocessing import StandardScaler
# 离线检验
# 使用方法：决策树集成回归 + 时间窗口法
# 预测结果，给出预测数据，预测步数，得到预测结果,全部结果
# data为一维向量
# 为了保证正常运行，需要先对数据进行归一化处理，然后再返回
def getPredictResultWithSlidingWindows(data):
    data = data.ravel().reshape(-1,1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # 上面完成归一化
    ratio = 0.9 #测试数据为10%
    window_size = 7
    X_train = getWindow(data[:int(len(data)*ratio)],window_size)
    y_train = data[window_size:int(len(data)*ratio)+1]
    #param_test = {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}
    #svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,param_grid=param_test)
    #svr.fit(X_train,y_train.ravel())
    #print_best_score(svr,param_test)
    svr = SVR(kernel='rbf',gamma='scale')
    svr.fit(X_train,y_train.ravel())

    X_test = getWindow(data[int(len(data)*ratio)+1 - window_size:-1],window_size)
    y_test = data[int(len(data)*ratio)+1:]
    y_prediction = svr.predict(X_test)
    y_test = scaler.inverse_transform(y_test)
    y_prediction = scaler.inverse_transform(y_prediction)
    return y_test,y_prediction

s = time.time()
window_size = 7
for i in range(len(store)):
    y_test,y_prediction = getPredictResultWithSlidingWindows(store[i])
    store[i] = y_prediction
e = time.time()
print('predict time:',e-s,'s')

#################################################################
# 第五步：回归。将预测数据倒回原始数据进行回归

# 将数据与原始数据进行对比，同时完成归一化

from sklearn.metrics import mean_squared_error
score = 0
ratioScore = 0

predictResult = []
for k in range(len(formatted_dataset)):
    label = secondClusans[k]
    repres = store[int(label)]
    m = repres * np.sqrt(np.var(originStdData[k])) + np.mean(originStdData[k])
    predictAns = m * np.sqrt(np.var(formatted_dataset[k])) + np.mean(formatted_dataset[k])
    predictResult.append(predictAns)
    data = formatted_dataset[k]
    mse = mean_squared_error(data[int(ratio*len(data))+1:],predictAns)
    score += mse
    ratioScore += mse/np.mean(formatted_dataset[k])
print(score/len(formatted_dataset))
print(ratioScore/len(formatted_dataset))

# 对照组：对原始时间序列直接进行预测，两者的差距
oppo = []
score = 0
ratioScore = 0

s = time.time()
window_size = 7
for i in range(len(formatted_dataset)):
    y_test,y_prediction = getPredictResultWithSlidingWindows(formatted_dataset[i])
    oppo.append(y_prediction)
    data = formatted_dataset[i]
    mse = mean_squared_error(data[int(ratio*len(data))+1:],y_prediction)
    score += mse
    ratioScore += mse/np.mean(formatted_dataset[i])
e = time.time()
print('predict time:',e-s,'s')
print(score/len(formatted_dataset))
print(ratioScore/len(formatted_dataset))

######################################################
# 模拟。对于每一个测试数据，构造预测器，并且模拟预测数据
# 模拟器：读入真实数据与预测数据，并进行线性插值，单位为分钟。容器从拉起到分配流量时间定为1分钟，调度器以分钟为单位进行调度。

# 主程序：模拟器。
# 输入：两端时间序列（真实值与预测值），长度为48个点，每个点一个小时
# 输出：模拟的响应时间序列
# 假设：存在一个理想的负载平衡器，能够平均将所有流量分配给每一个容器，且最大平均流量不超过150req/s（超过的则拒绝服务）
# 1. 容器从进行调度到负载流量需要1分钟。
# 2. 每分钟进行一次调度判断操作

# 辅助程序：调度器
# 根据预测数据判断是否进行调度：首先看预测数据与实际数据的差值，如果差值足够小，则按照预测器数据进行。如果差值过大，则将预测器的变动作为参考（拐点预测器），以阈值法进行调度。如果拐点预测继续失败，则回归原始预测器
# 需要注意：流量只可能是整数，需要进行取整操作

#本步骤的结果会反过来影响前面的结论，请慎重进行



# 得到线性插值结果，每个点插60个值。2个点61,3个点121
def getLine(line):
    result = np.zeros((len(line)-1)*5+1)
    for i in range(len(line)):
        if i==0:
            result[0] = line[0]
        else:
            result[i*5] = line[i]
            #进行插值
            delta = (line[i]-line[i-1])/5
            for j in range(59): #对中间的59个点进行插值处理
                result[(i-1)*5+j+1] = result[(i-1)*5]+delta * (j+1)
    return result

# 根据流量得到响应时间的函数
def getResponseTime(traffic):
    traffic = int(traffic)
    data = [0,10,29,42,62,85,140,168,209,250,269,283,329,378,390,467,579]
    if traffic > 150:
        return 1500
    else:
        if traffic % 10 ==0:
            return data[traffic//10]
        else:
            num = traffic//10
            delta = (data[num+1]-data[num])/10
            d = traffic - num*10
            return data[num] + d*delta

# 预测算法，关键是预测趋势和拐点。
# 取得趋势
def getTrend(data):
    if data[2]>data[1] and data[1]>data[0]:
        return 1
    elif data[2]<data[1] and data[1]<data[0]:
        return -1
    return 0


# 根据过去和现在的真实数据和预测数据预测下一分钟的容器数量
# real的为之前的观测值，pred为之前的预测值加上当前的预测值
# 对于5以内较强的性质是可以跑出来的，v>0.6,p<0.2，且之前的斜率比较符合，就可以直接跑出结果。
# 如果之前的结果不太符合，那就做线性映射

# 对于减少的容器数量需要慎重的考虑，因为可能会导致SLA超时
# 产生一个列表来存储减少的容器数量，如果连续三次都是减少，则执行，返回的是减少的平均数
def getContainer(real,pred,responseTime,containerNum):
    #如果预测的流量过小，直接转为阈值法
    # 如果预测的流量不足三个容器，直接忽视
    if real[-1]<200 or len(real)<5:
        if responseTime > 250:
            containerNum += 1
        elif responseTime < 50:
            containerNum -= 1
        if containerNum < 1:
            containerNum = 1
        return containerNum,False

    # 如果当前已经超时，则直接进行阈值法调度
    if responseTime > 250:
        containerNum += 1
        return containerNum,False
    # 如果当前的预测值与真实值很接近，则直接按照预测值预测
    # 如果当前的预测值与真实值不接近，但是斜率接近，则按照预测值预测斜率
    # 如果都不接近，则转为阈值法调度器，直接按照响应时间预测。对于超过200的响应时间直接+1
    delta = 50
    # 基于精度的减少需要前面两个点都减少了才能减
    if abs(real[-1]-pred[-2])<delta:#如果在范围以内。可能性很小，适用于拟合的比较好的情况（比如单个的预测）
        p = math.ceil(pred[-1]/80)
        if p < 1:
            p = 1
        # 如果当前p值需要减少
        if p < containerNum:
            #检验之前需要持续下降，且两个点都预测准确，
            if real[-1]<real[-2] and real[-2]<real[-3] and abs(real[-2]-pred[-3])<delta:
                return p,True
            else:
                return containerNum,True
        return p,True

    # 使用spearman进行趋势判断
    # 对于v>0.6且p<0.2的结果，认为是同趋势
    # 对于同趋势的结果，分两种情况。分别比较最近的三个点的两个差值，如果都相同，那就直接预测
    # 如果不相同且整体相差一个比例，那就按照比例预测
    # 如果不相同且不按照比例，那就加权预测

    # 如果长度不够5，则直接忽略
    # 比例减少这边直接减
    if len(real)>=5:
        v,p = spearmanr(real[-5:],pred[-6:-1])
        if v>0.6 and p<0.2:
            r1 = real[-1]-real[-2]
            r2 = real[-2]-real[-3]
            p1 = pred[-2]-pred[-3]
            p2 = pred[-3]-pred[-4]
            #比例计算
            delta = np.array([abs((r1-p1)/r1),abs((r1-p1)/p1),abs((r2-p2)/r2),abs((r2-p2)/p2)])
            ratio1 = r1/p1
            ratio2 = r2/p2
            if abs(ratio1) < 2 and abs(ratio2)<2: #预测的趋势很相近，可以直接沿用。具体表现为比例相近
                #直接使用
                futureTraffic = real[-1] + (pred[-1]-pred[-2])
                p = math.ceil(futureTraffic/80)
                if p<1:
                    p=1
                return p,True
            elif abs(ratio1)<1 or (sum(delta>2)<2):# 尽管比例不相近，但是整体趋势接近。预测方式是综合判断，用过去两个点的值来预测下一个点
                #判断比例
                # 使用实际值与预测值的插值比例来进行预测
                avgDiff1 = 0.5*r1+0.5*r2
                avgDiff2 = 0.5*p1+0.5*p2
                newDiff = 0.6*(pred[-1]-pred[-2])+0.3*p1+0.1*p2
                futureTraffic = real[-1] + (newDiff/avgDiff2*avgDiff1)# 说实话，效果差强人意
                p = math.ceil(futureTraffic/80)
                if p<1:
                    p=1
                return p,True
                    
    # 阈值法只根据当前是否超时/过低来判断是否需要增减
    if responseTime > 250:
        containerNum += 1
    elif responseTime < 50:
        containerNum -= 1
    if containerNum < 1:
        containerNum = 1
    return containerNum,False

#判断预测器是否进行了预测
def isGetPrediction(real,pred):
    # 如果当前的预测值与真实值很接近，则直接按照预测值预测
    # 如果当前的预测值与真实值不接近，但是斜率接近，则按照预测值预测斜率
    # 如果都不接近，则转为阈值法调度器，直接按照响应时间预测。对于超过200的响应时间直接+1
    delta = 80
    if abs(real[-1]-pred[-2])<delta:#如果在范围以内。可能性很小，适用于拟合的比较好的情况（比如单个的预测）
        p = pred[-1]//80
        if p < 1:
            p = 1
        return True
    
    # 后续，提取趋势进行预测
    # 如何进行趋势判断，我认为还是要取斜率。
    # 如果三点同趋势，
        # 趋势与真实数据趋势相同：将三点斜率转成角度取加权平均后再转成斜率进行预测
    if len(real)>=3:
        s1 = getTrend(real[-3:])
        s2 = getTrend(pred[-4:-1])
        if s1 == s2:# 趋势相同
            realDelta = 0.66*abs(real[-1]-real[-2]) + 0.33*abs(real[-2]-real[-3])
            predDelta = 0.66*abs(pred[-2]-pred[-3]) + 0.33*abs(pred[-3]-pred[-4])
            if abs(predDelta-realDelta)/realDelta < 0.5: #误差在一定范围以内，直接预测
                return True
    return False
                    
#根据所给的数据，返回单纯使用响应式调度所得到的时间
def reactionSimu(data):
    response = []
    conNum = []
    simu_real = data#getLine(data)#将小时平均到分钟，并进行插值
    containerNum = int(simu_real[0]/80)#初始容器数
    futureNum = int(simu_real[0]/80)
    if containerNum < 1:
        containerNum = 1
    for i in range(len(simu_real)):# 进行模拟，单位为分钟
        # 注意：第i时刻的真实数据要在迭代之后才能用
        # 记录当前的响应时间
        res = getResponseTime(simu_real[i]/containerNum)
        response.append(res)
        conNum.append(containerNum)
        containerNum = futureNum # 上一分钟的数据调度已经调度过来了
        if containerNum < 1:
            containerNum = 1
        # 调度器，根据过去和现在数据预测下一分钟

        if res > 250:
            futureNum = containerNum + 1
        elif res < 50:
            futureNum = containerNum - 1
    return np.array(response),np.array(conNum)

#拿到两条时间序列，返回响应时间序列
def simulation(data,predict):
    predNum = 0
    response = []
    conNum = []
    simu_real = data#getLine(data)#将小时平均到分钟，并进行插值
    simu_pred = predict#getLine(predict)
    containerNum = int(simu_real[0]/80)#初始容器数
    futureNum = int(simu_real[0]/80)
    # 需要有三个值，一个记录上一次是否超时，一个记录上次的措施，一个记录惩罚值。
    # 在惩罚值为归零前，直接使用阈值法判断
    for i in range(len(simu_real)):# 进行模拟，单位为分钟
        # 注意：第i时刻的真实数据要在迭代之后才能用
        # 记录当前的响应时间
        if containerNum < 1:
            containerNum = 1
        res = getResponseTime(simu_real[i]/containerNum)
        response.append(res)
        conNum.append(containerNum)
        containerNum = futureNum # 上一分钟的数据调度已经调度过来了
        # 调度器，根据过去和现在数据预测下一分钟
        if i == len(simu_real)-1:
            break
        else:
            futureNum,getPredict = getContainer(simu_real[:i+1],simu_pred[:i+2],res,containerNum)
            if getPredict:
                predNum += 1
    return np.array(response),np.array(conNum),predNum

### 开始模拟
s = time.time()
total = 0
p1 = []
p2 = []
p3 = []
con1 = []
con2 = []
con3 = []
#开始进行测试。测试的结果有两个，看集成预测结果与上界的逼近情况与下界的疏离情况。
# 评判标准：SLA违约点数，以及平均消耗资源量。以最少的资源满足SLA为最优（SLA违约点最小）
for i in range(len(formatted_dataset)):
    #首先检查流量情况，如果流量平均小于100，没有必要
    predictOne = predictResult[i] #集成预测结果
    predictTwo = oppo[i]          #单独预测结果
    data = formatted_dataset[i]   # 真实数据
    realOne = data[int(ratio*len(data))+1:]
    realOne = realOne.ravel()
    if sum(realOne)/len(realOne) < 100:
        p1.append(0)
        p2.append(0)
        p3.append(0)
        con1.append(1)
        con2.append(1)
        con3.append(1)
    else:
        r1,c1 = simulation(realOne,predictOne)
        r2,c2 = simulation(realOne,predictTwo)
        r3,c3 = reactionSimu(realOne)
        p1.append(sum(r1>250))
        p2.append(sum(r2>250))
        p3.append(sum(r3>250))
        con1.append(sum(c1)/len(c1))
        con2.append(sum(c2)/len(c2))
        con3.append(sum(c3)/len(c3))

plt.plot(p1,color='red')
plt.plot(p2,color='black')
plt.plot(p3,color='blue')
plt.show()
plt.plot(con1,color='red')
plt.plot(con2,color='black')
plt.plot(con3,color='blue')
plt.show()
e = time.time()
print(e-s,'s')
p1 = np.array(p1)
p2 = np.array(p2)
p3 = np.array(p3)
con1 = np.array(con1)
con2 = np.array(con2)
con3 = np.array(con3)
delta1 = p1-p2
delta2 = p1-p3
print(sum(delta1))
print(sum(delta2))
print(sum(np.array(p1)))
print('违约情况比较：')
print('集成预测自身总违约时间：',sum(p1))
print('集成预测比单纯预测少的时间：',sum(delta1))
print('集成预测比阈值法多的时间：',sum(delta2))
print('集成预测：',sum(con1))
print('单纯预测：',sum(con2))
print('阈值法：',sum(con3))

#评价部分
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import math
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
            s.append(mean_squared_error(center,stdData[index]))
    
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

evaluateResult(y_pre,20)

#探索聚类在最后预测部分的点的表现
def evaluateClusPredict(clusResult,bigClusThre = 20):
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
            l.append(spearmanr(center[-47:],stdData[index,-47:])[0])
            s.append(mean_squared_error(center[-47:],stdData[index,-47:]))
    
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

