# 这里只使用趋势进行聚类，只使用趋势进行预测

# 使用spearmanr系数进行聚类。p值不小于0.5的直接设置为0。假设得到的为x，则转为e^(1/x)作为距离

# 问题：如何判断趋势是否正确？一阶差分！

# 具体的实现：根据残差数据进行聚类
# 直接进行预测
# 下一阶段目标：查看不使用滑动窗口法对预测结果的影响有多大
# 对同一段残差流量，一个使用滑动窗口法，一个不使用滑动窗口法，比较

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

####聚类相关函数#################
from scipy.stats import spearmanr
import math
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

from sklearn.metrics import pairwise_distances
def trend_affinity(X):
    return pairwise_distances(X, metric=getTrendScore)
# 聚类方式：1. 只对原始数据进行聚类。2.只对残差数据聚类。3.即对原始数据一聚类，残差数据二聚类。4.残差数据一聚类，原始数据二聚类（Kmeans）
# 衡量标准
# 加速：考虑到层次聚类速度很慢（要生成距离矩阵），可以自行实现层次聚类的结果
# 第一步：初步筛查。由于聚类本身的特性，所以有很多的数据其实根本就不需要考虑。
# 直接从每个元素开始对每个元素进行遍历，将其分为多个集合。（已经在一个集合里面的不用访问）所有能访问到的元素都放在一个集合里面。
# 然后再进行层次聚类 可以考虑自己实现
# 问题：没有本质区别
# 加速想法二：使用rank-base对一天的数据进行处理后直接Kmeans分100类，再对每一类进行区间聚类
#################################################################
# 初始，根据原始数据计算新数据
# 从paa这里

# 需要在聚类前将训练数据划分完毕。ratio必须要精心选择使得paa的值能够成为整数

ratio = 0.9
n_paa_segments = 18
paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
originData = stdData[:,:int(ratio*stdData.shape[1])] #训练部分已经知道的原始数据
paa_mid = paa.fit_transform(originData)
paa_mid = paa_mid.reshape(paa_mid.shape[0],paa_mid.shape[1])
baseData = paa.inverse_transform(paa_mid)#提取基线数据
restData = originData - baseData # 计算得到残差数据

# 模式提取（直接加和取平均后求rank-base处理，或者再做标准化进行SAX处理）
# 初步想法：将每天24小时的流量重复叠加取平均，进行rank-base处理，然后跑MSE用Kmeans进行100聚类
# 想法二：在自己
# 对于100聚类中的每个聚类，再跑层次聚类进行细分，最小调到1。聚类结果衡量用类内最大相似度来进行衡量（有多不相近）
# 使用SAX的dayPattern


# 做法01：使用SAX提取前三天的残差信息，进行20聚类。对每个聚类内部跑complete，0.5的层次聚类。考虑到500量级要跑3分钟，平均大约是一个小时。
from sklearn.cluster import AgglomerativeClustering
import time

dayPattern = []
for index in range(restData.shape[0]):
    cuData = restData[index].ravel()
    day = len(cuData)//24
    total = np.zeros(24)
    for d in range(3):
        total += cuData[d*24:(d+1)*24]
    dayPattern.append(total/day)

dayPattern = np.array(dayPattern)
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
dayPattern = scaler.fit_transform(dayPattern)
n_paa_segments = 24
n_sax_symbols = 5
sax = SymbolicAggregateApproximation(n_segments=n_paa_segments,
                                     alphabet_size_avg=n_sax_symbols)
dayPattern = sax.fit_transform(dayPattern)
dayPattern = dayPattern.reshape(dayPattern.shape[0],dayPattern.shape[1])
#进行聚类
# 对SAX处理后的日变动进行50聚类
s = time.time()
y_pre = KMeans(n_clusters=20).fit_predict(dayPattern)
clusNum = np.zeros(len(y_pre))
totalClus = 0
for k in range(max(y_pre)+1):
    data = restData[y_pre==k]
    data = data.reshape(data.shape[0],data.shape[1])
    distance_matrix = trend_affinity(data)
    model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',distance_threshold = pow(math.e,1/0.5))
    second_pre = model.fit_predict(distance_matrix)
    #对于所有在该类别中的元素，进行加和
    second_iter = np.where(y_pre==k)[0]
    for index,ele in enumerate(second_iter):
        clusNum[ele] = second_pre[index]+totalClus
    totalClus += max(second_pre)+1

e = time.time()
print(e-s,'s')
## 后面的部分必须要去除基线数据。或者说，必须要以残差为基础进行聚类中心选取和预测
# 通过残差聚类得到中心
# 对残差滚动时间序列窗口得到预测结果
# 将预测结果加上最后的基线数据进行逆归一化
# 得到最终结果，指导变化
# 整合部分
totalClusterNum = totalClus
secondClusans = clusNum
# 第三步：整体预测。提取出聚类中心并对其进行预测，将结果作为聚类中所有值的最终结果
store = []
for k in range(totalClusterNum):
    stdClusData = restData[secondClusans == k]
    store.append(sum(stdClusData)/stdClusData.shape[0])

#################################################################
# 第四步：预测。
# 通过对过去432的点，预测未来后面的1个点+47个点
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# 返回滑动窗口
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


# 离线检验
# 使用方法：决策树集成回归 + 时间窗口法
# 预测结果，给出预测数据，预测步数，得到预测结果,全部结果
def getPredictResultWithSlidingWindows(data):
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
    return y_test,y_prediction
# data为一维向量
def futurePredict(data):
    ratio = 0.9 #测试数据为10%
    window_size = 7
    X_train = getWindow(data[:-1],window_size)
    y_train = data[window_size:]
    #param_test = {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}
    #svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,param_grid=param_test)
    #svr.fit(X_train,y_train.ravel())
    #print_best_score(svr,param_test)
    svr = SVR(kernel='rbf',gamma='scale')
    svr.fit(X_train,y_train.ravel())
    # 后面使用叠加法反复叠加出来，需要仔细调试
    y_prediction = np.zeros(48)
    window = np.zeros(window_size)
    # 从最后的window_size个元素中装填预测窗口
    for i in range(window_size):
        window[i] = data[-1*(window_size-i)]

    for i in range(len(y_prediction)):
        p = window.reshape(1,window_size)
        y_prediction[i] = svr.predict(p)[0]
        window = np.roll(window,-1)#选择往左移动一步
        window[window_size-1] = y_prediction[i]
    return y_prediction[1:]

s = time.time()
window_size = 7
for i in range(len(store)):
    y_prediction = futurePredict(store[i])#这里输入为432个点，输出为47个点
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
    repres = store[int(label)] + baseData[k,-1][0]#残差数据+最后的基线数据
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
    mse = mean_squared_error(data[int(ratio*len(data))+1:],predictAns)
    score += mse
    ratioScore += mse/np.mean(formatted_dataset[i])
e = time.time()
print('predict time:',e-s,'s')
print(score/len(formatted_dataset))
print(ratioScore/len(formatted_dataset))

# 最后的模拟器
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
    result = np.zeros((len(line)-1)*60+1)
    for i in range(len(line)):
        if i==0:
            result[0] = line[0]
        else:
            result[i*60] = line[i]
            #进行插值
            delta = (line[i]-line[i-1])/60
            for j in range(59): #对中间的59个点进行插值处理
                result[(i-1)*60+j+1] = result[(i-1)*60]+delta * (j+1)
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
def getContainer(real,pred,responseTime,containerNum):
    # 如果当前的预测值与真实值很接近，则直接按照预测值预测
    # 如果当前的预测值与真实值不接近，但是斜率接近，则按照预测值预测斜率
    # 如果都不接近，则转为阈值法调度器，直接按照响应时间预测。对于超过200的响应时间直接+1
    delta = 80
    if abs(real[-1]-pred[-2])<delta:#如果在范围以内。可能性很小，适用于拟合的比较好的情况（比如单个的预测）
        p = pred[-1]//80
        if p < 1:
            p = 1
        return p
    
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
                if abs(s1)>0:#不是拐点，则依据前面的值加权处理
                    futureTraffic = real[-1]+0.5*(pred[-1]-pred[-2])+0.5*(pred[-2]-pred[-3])
                    p = futureTraffic//80
                    if p<1:
                        p=1
                    return p
                else: #如果是拐点，则按照拐点进行预测。不直接枚举预测下降是为了避免预测错误
                    futureTraffic = real[-1]+(pred[-1]-pred[-2])
                    p = futureTraffic//80
                    if p<1:
                        p=1
                    return p
                    
    # 阈值法只根据当前是否超时/过低来判断是否需要增减
    if responseTime > 250:
        containerNum += 1
    elif responseTime < 100:
        containerNum -= 1
    if containerNum < 1:
        containerNum = 1
    return containerNum

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
    simu_real = getLine(data/60)#将小时平均到分钟，并进行插值
    containerNum = int(simu_real[0]/80)#初始容器数
    futureNum = int(simu_real[0]/80)
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
        if res > 250:
            futureNum = containerNum + 1
        elif res < 50:
            futureNum = containerNum - 1
    return np.array(response),np.array(conNum)


#拿到两条时间序列，返回响应时间序列
def simulation(data,predict):
    response = []
    conNum = []
    simu_real = getLine(data/60)#将小时平均到分钟，并进行插值
    simu_pred = getLine(predict/60)
    containerNum = int(simu_real[0]/80)#初始容器数
    futureNum = int(simu_real[0]/80)
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
            futureNum = getContainer(simu_real[:i+1],simu_pred[:i+2],res,containerNum)
    return np.array(response),np.array(conNum)

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
        continue
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

# 实验滑动窗口的影响
# 用来训练的部分数据
paa_origin = paa.inverse_transform(paa_mid)
paa_rest = stdData[:,:int(ratio*stdData.shape[1])]-paa_origin

def futurePredict(data):
    ratio = 0.9 #测试数据为10%
    window_size = 7
    X_train = getWindow(data[:-1],window_size)
    y_train = data[window_size:]
    #param_test = {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}
    #svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,param_grid=param_test)
    #svr.fit(X_train,y_train.ravel())
    #print_best_score(svr,param_test)
    svr = SVR(kernel='rbf',gamma='scale')
    svr.fit(X_train,y_train.ravel())
    # 后面使用叠加法反复叠加出来，需要仔细调试
    y_prediction = np.zeros(48)
    window = np.zeros(window_size)
    # 从最后的window_size个元素中装填预测窗口
    for i in range(window_size):
        window[i] = data[-1*(window_size-i)]

    for i in range(len(y_prediction)):
        p = window.reshape(1,window_size)
        y_prediction[i] = svr.predict(p)[0]
        window = np.roll(window,-1)#选择往左移动一步
        window[window_size-1] = y_prediction[i]
    return y_prediction[1:]

# 改进：
# 导入真实数据
# 将真实数据减去第一天作为残差的输入
# 使用滑动窗口法
# 注：data导入的是全体流量。
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

# 将残余数据的聚类中心放入store中
store = []
for k in range(totalClusterNum):
    # 残余数据进行集中聚类
    # 由两个部分组成，已知部分由计算出来的残余数据组成，未知部分由实际数据减去第一个点组成
    calc = np.zeros(len(stdData[k]))
    for i in range(int(ratio*stdData.shape[1])):
        calc[i] = paa_rest[k,i,0]
    for i in range(int(ratio*stdData.shape[1]),len(stdData[k])):
        calc[i] = stdData[k,i] - paa_mid[k,-1]
    store.append(calc)
    # 
    