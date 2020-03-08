# Birch论文 A BIRCH-Based Clustering Method for Large Time Series Databases
# 固定读取部分
import time
foreignTime = time.time()
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
DistAvg = 300
VarAvg = 5000
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
#y_pre = KMeans(n_clusters = max(y_pre)+1,random_state = 0).fit_predict(data)
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


import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras import optimizers
# 给出训练数据，返回训练好的模型
def getLSTMModel(main_data,aux_data,y_train,epch):
    main_input = Input(shape=(1,24), dtype='float32', name='main_input')# 输入的连续时间戳长度为24
    #x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
    lstm_out = LSTM(24,dropout = 0.5,return_sequences = True)(main_input)
    lstm_out = LSTM(24)(lstm_out)
    auxiliary_input = Input(shape=(26,),name='aux_input')
    x = keras.layers.concatenate([lstm_out, auxiliary_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    main_output = Dense(1, activation='linear', name='main_output')(x)
    model = Model(inputs=[main_input,auxiliary_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='mean_squared_error',loss_weights=[1.])
    model.fit([main_data,aux_data], [y_train],epochs=epch, batch_size=20)
    return model

# 问题：预测的结果无法小于0
# 猜测是辅助函数这边的问题
# 构造一个普通的LSTM
def getLSTMResult(data,epch=15):
    #第一步，切分数据集
    ratio = 0.9
    #然后对于每个24小时进行训练。用前24小时去精确预测后24小时中的某一个点。
    # 对训练数据的24小时内生成训练窗口集合。
    # 内部时间戳，24个连续的时间戳
    # 外部辅助特征，23个差分+平均值+方差+1天中的第几个小时
    s = time.time()
    st = []
    for index in range(len(data)):
        # 对于每一个时间序列数据
        main_input_list = []
        auxiliary_input_list = []
        y_train = []
        # 对于每一个点，将其前面的24个点打包成连续时间戳
        # 提取前24个点的均值和方差，以及23个一阶差分数据
        # 以及这个点是该天的第几个小时
        window_size = 24
        scaler = StandardScaler()
        tsData = scaler.fit_transform(data[index])
        for i in range(window_size,int(len(tsData)*ratio)): #需要预测的时间点
            main_input_list.append(tsData[i-window_size:i]) #放入连续的24个时间戳
            aux = np.zeros(26)
            aux[0] = i%24
            aux[1] = np.mean(tsData[i-window_size:i])
            aux[2] = np.var(tsData[i-window_size:i])
            aux[3:] = np.diff(tsData[i-window_size:i].ravel())
            auxiliary_input_list.append(aux)
            y_train.append(tsData[i])
        main_input_list = np.array(main_input_list)
        main_input_list = main_input_list.reshape(main_input_list.shape[0],1,main_input_list.shape[1]) #变成数量*维数*样本数的形式
        auxiliary_input_list = np.array(auxiliary_input_list)
        y_train = np.array(y_train)
        model = getLSTMModel(main_input_list,auxiliary_input_list,y_train,epch)

        #生成测试数据  
        #对未来的47个点，生成对应的预测值
        test_input = []
        aux_input = []
        for i in range(int(len(tsData)*ratio)+1,len(tsData)):
            test_input.append(tsData[i-window_size:i]) #放入连续的24个时间戳
            aux = np.zeros(26)
            aux[0] = i%24
            aux[1] = np.mean(tsData[i-window_size:i])
            aux[2] = np.var(tsData[i-window_size:i])
            aux[3:] = np.diff(tsData[i-window_size:i].ravel())
            aux_input.append(aux)

        test_input = np.array(test_input)
        test_input = test_input.reshape(test_input.shape[0],1,test_input.shape[1]) #变成数量*维数*样本数的形式
        aux_input = np.array(aux_input)
        y_test = model.predict({'main_input':test_input,'aux_input':aux_input})
        st.append(scaler.inverse_transform(y_test))
        del model
    e = time.time()
    print('lstm predict time:',e-s,'s')
    return st


'''
#选择用函数
for i in range(9900):
    if max(formatted_dataset[i,int(ratio*len(data))+1:]) < 520 and max(formatted_dataset[i,int(ratio*len(data))+1:]) - min(formatted_dataset[i,int(ratio*len(data))+1:]) > 300:
        print(i)
        plt.plot(formatted_dataset[i,int(ratio*len(data))+1:],color='red',label="real")
        plt.plot(oppo[i],color='green',label="solo")
        plt.plot(predictResult[i],color='blue',label="clus")
        plt.legend(loc='upper right')
        plt.show()
        print('单独误差：',mean_squared_error(formatted_dataset[i,int(ratio*len(data))+1:],oppo[i]))
        print('集成误差：',mean_squared_error(formatted_dataset[i,int(ratio*len(data))+1:],predictResult[i]))

'''
# 表现较好的：90/4931
# 可能较好：2053，7332
# 单独预测表现更好：1423，5184
# 集成预测表现更好：1024，5104
# 都不好：1289，2489争取不被压爆
index = 4931 #用于测试的数据编号。候补，
k = index
label = int(secondClusans[k])
#y_test,y_prediction = getPredictResultWithSlidingWindows(store[index])
y_prediction = getLSTMResult([store[int(label)]])
y_prediction = np.array(y_prediction)
y_prediction = y_prediction.reshape(47)
from sklearn.metrics import mean_squared_error
k = index
repres = y_prediction
m = repres * np.sqrt(np.var(originStdData[k])) + np.mean(originStdData[k])
predictAns = m * np.sqrt(np.var(formatted_dataset[k])) + np.mean(formatted_dataset[k])
data = formatted_dataset[k]
print(mean_squared_error(data[int(ratio*len(data))+1:],predictAns))

y_prediction_solo = getLSTMResult([stdData[index]])
y_prediction_solo = np.array(y_prediction_solo)
y_prediction_solo = y_prediction_solo.reshape(47)
k = index
data = formatted_dataset[k]
repres = y_prediction_solo
m = repres * np.sqrt(np.var(originStdData[k])) + np.mean(originStdData[k])
y_prediction_solo = m * np.sqrt(np.var(formatted_dataset[k])) + np.mean(formatted_dataset[k])
print(mean_squared_error(data[int(ratio*len(data))+1:],y_prediction_solo))

def WriteTotalTrafficToFile(data):# 时序数据，包括第一个点+47个预测点
    fileName = 'trafficTotal.txt'
    dataPath = "E:\\code\\myPaper\\k8sPredictor"
    LocalPath = os.path.join(dataPath,fileName)
    writer = open(LocalPath,'w',encoding='utf-8')
    for ele in data:
        writer.write(str(ele)+'\n')

def WriteSpecTrafficToFile(data):# 时序数据，包括第一个点+47个预测点
    fileName = 'trafficSpec.txt'
    dataPath = "E:\\code\\myPaper\\k8sPredictor"
    LocalPath = os.path.join(dataPath,fileName)
    writer = open(LocalPath,'w',encoding='utf-8')
    for ele in data:
        writer.write(str(ele)+'\n')

# 将流量数据写入给gatling
def WriteTestTrafficToFile(data):
    fileName = 'wtytest'
    dataPath = "E:\\code\\myPaper\\k8sPredictor"
    LocalPath = os.path.join(dataPath,fileName)
    writer = open(LocalPath,'w',encoding='utf-8')
    for ele in data:
        writer.write(str(ele)+'\n')

writeCache = np.zeros(48)
writeCache[0] = formatted_dataset[index,432,0]
writeCache[1:] = predictAns
WriteTotalTrafficToFile(writeCache)
writeCache = np.zeros(48)
writeCache[0] = formatted_dataset[index,432,0]
writeCache[1:] = y_prediction_solo
WriteSpecTrafficToFile(writeCache)
WriteTestTrafficToFile(formatted_dataset[index,432:].ravel())

import matplotlib
matplotlib.use('qt4agg')
#指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

# 对照组：对原始时间序列直接进行预测，两者的差距
plt.plot(data[int(ratio*len(data))+1:],color='red',label="真实数据")
plt.plot(y_prediction_solo,color='green',label="单独预测")
plt.plot(predictAns,color='blue',label="集成预测")
plt.legend(loc='upper right')
plt.savefig('data'+str(index)+'.png')
print(foreignTime-time.time(),'s')