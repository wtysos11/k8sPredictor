# 实现预测式调度器的对比
# ARMA策略：使用ARMA模型选择最佳点进行预测
# 线性回归策略

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
DistAvg = 1000
VarAvg = 2000
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
from tslearn.preprocessing import TimeSeriesScalerMinMaxpl
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
originStdData = stdData.copy()
stdData = scaler.fit_transform(stdData)

# 进行机器学习预测
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import numpy as np
import time
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

# 由于该预测算法是实时的算法，分解预测难以进行
# 一旦分解预测就无法获取最新的值，可以考虑基线分解预测和小波变化分解预测

# 传递完整数据data，预测10%的尾部数据

# 需要加上1. XGBoost等集成树算法 2. 需要对窗口进行适当的枚举
# data传入的是时序数*特征数*1
def svrPredict(data):
    s = time.time()
    store = []
    for i in range(len(data)):
        y_test,y_prediction = getPredictResultWithSlidingWindows(data[i])
        store.append(y_prediction)
    e = time.time()
    print('svr predict time:',e-s,'s')
    return store

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
import time
# nerual network，简单的前向神经网络
# 自行设计

# Keras LSTM算法，这里使用的是下一个方法
# 未经过测试
def LSTMPredict(data):
    s = time.time()
    store = []
    for i in range(len(data)):
        window_size = 7
        # 首先对data中的前面进行训练
        X_train = data[i,:int(len(data[i])*ratio)] # 目标训练数据
        X_train = getWindow(X_train,window_size)
        y_train = data[i,window_size:int(len(data[i])*ratio)+1]
        X_test = data[i,int(len(data[i])*ratio)+1-window_size:-1]# 应该有47个
        X_test = getWindow(X_test,window_size)

        # 主程序部分：
        hidden_size = 3
        output_size = 1
        batch_size = 8
        epoch_time = 10
        X_train = X_train.reshape(len(X_train),1,window_size)
        regressor = Sequential()
        regressor.add(LSTM(hidden_size, return_sequences=True, input_shape=(X_train.shape[1], window_size)))
        regressor.add(LSTM(hidden_size, return_sequences=True))
        regressor.add(LSTM(hidden_size))
        regressor.add(Dense(output_size))
        regressor.compile(loss='mean_squared_error',optimizer='adam')
        regressor.fit(X_train, y_train, batch_size = batch_size, epochs = epoch_time, verbose = 0.2,shuffle=False)
        
        inputs = X_test
        inputs = np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1]))
        y_pred = regressor.predict(inputs)
        store.append(y_pred)
        # 然后对后面48个中的前47个进行测试
        # 将数据导回
    e = time.time()
    print('lstm predict time:',e-s,'s')

# 直接使用简单RNN
# 平均每个时间序列5s
def RNNPredict(data):
    s = time.time()
    store = []
    for i in range(len(data)):
        window_size = 7
        # 首先对data中的前面进行训练
        X_train = data[i,:int(len(data[i])*ratio)] # 目标训练数据
        X_train = getWindow(X_train,window_size)
        y_train = data[i,window_size:int(len(data[i])*ratio)+1]
        X_test = data[i,int(len(data[i])*ratio)+1-window_size:-1]# 应该有47个
        X_test = getWindow(X_test,window_size)
        # 主程序部分：
        hidden_size = 3
        output_size = 1
        batch_size = 8
        epoch_time = 10
        X_train = X_train.reshape(len(X_train),1,window_size)
        regressor = Sequential()
        regressor.add(SimpleRNN(hidden_size, return_sequences=True, input_shape=(X_train.shape[1], window_size)))
        regressor.add(SimpleRNN(hidden_size, return_sequences=True))
        regressor.add(SimpleRNN(hidden_size))
        regressor.add(Dense(output_size))
        regressor.compile(loss='mean_squared_error',optimizer='adam')
        regressor.fit(X_train, y_train, batch_size = batch_size, epochs = epoch_time, verbose = 0.2,shuffle=False)

        inputs = X_test
        inputs = np.reshape(inputs,(inputs.shape[0],1,inputs.shape[1]))
        y_pred = regressor.predict(inputs)
        store.append(y_pred)
        # 然后对后面48个中的前47个进行测试
        # 将数据导回
    e = time.time()
    print('rnn predict time:',e-s,'s')

# 贝叶斯岭回归BRR

# arima
# 如果不inevitable，尝试进行差分
# 不然直接放弃
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA,ARMA
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
def armaPredict(data):
    ratio = 0.9
    s = time.time()
    # 开始的时候进行两次差分，选择最平稳的数据进入
    result = []
    for i in range(len(data)):
        print('loop',i)
        try:
            train_data = data[i,:int(ratio*len(data[i]))].ravel()
            order = sm.tsa.arma_order_select_ic(train_data,ic='aic')['aic_min_order']
            #order = (4,2)
            # 结合生成ARIMA模型
            model = ARMA(train_data,order = order)
            res = model.fit()
            # 这一步值得商榷，未知arma使用什么来获得最后的结果
            pre = _arma_predict_out_of_sample(res.params, 48, res.resid, res.k_ar, res.k_ma, res.k_trend, res.k_exog, endog=train_data, exog=None, start=len(train_data))
            result.append(pre[1:])
            # 有可能中断，这时候尝试一阶差分，不然直接转线性
        except:
            #可能因为各种原因发生错误，这时候需要使用简单直接预测
            # 根据上一个点的差分来预测下一个点
            pre = np.zeros(len(data[i]) - int(len(data[i])*ratio) - 1)
            for j in range(len(pre)):
                index = int(len(data[i])*ratio)+j
                pre[j] = data[i,index,0] + (data[i,index,0] - data[i,index-1,0])#第i个点的预测值由上一个点的真实值加上上一个点与上两个点的差分
        
    e = time.time()
    print('predict time:',e-s,'s')
    return result

# 超出范围的预测




# 小波变化分解预测
def waveDecompose(data):
    import numpy as np
    from matplotlib import pyplot as plt
    import pandas as pd
    import pywt
    import statsmodels.api as sm
    from statsmodels.tsa.ar_model import AR
    from statsmodels.tsa.arima_model import ARIMA,ARMA
    from sklearn.preprocessing import MinMaxScaler

    split_factor = 0.8
    split_num = int(len(data_list[0])*split_factor)
    cols = train.columns[1:-1]

    data = np.array(y,'f')
    #进行切分
    X_train = data[:split_num]
    y_train = data[1:split_num+1]
    X_test = data[split_num:-1]
    y_test = data[split_num+1:]
    xsc = MinMaxScaler()
    ysc = MinMaxScaler()
    X_train = np.reshape(X_train,(-1,1))
    y_train = np.reshape(y_train,(-1,1))
    X_train = xsc.fit_transform(X_train)
    y_train = ysc.fit_transform(y_train)
    X_train = np.reshape(X_train,(-1))
    y_train = np.reshape(y_train,(-1))
    A2,D2,D1 = pywt.wavedec(X_train,'db4',mode='sym',level=2)

    order_A2 = sm.tsa.arma_order_select_ic(A2,ic='aic')['aic_min_order']
    order_D2 = sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']
    order_D1 = sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']

    model_A2 = ARMA(A2,order = order_A2)
    model_D2 = ARMA(D2,order = order_D2)
    model_D1 = ARMA(D1,order = order_D1)

    results_A2 = model_A2.fit()
    results_D2 = model_D2.fit()
    results_D1 = model_D1.fit()
    #输出信号分解后的拟合曲线
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(A2,color = 'blue')
    plt.plot(results_A2.fittedvalues,color = 'red')
    plt.title('model_A2')
    plt.subplot(3,1,2)
    plt.plot(D2,color = 'blue')
    plt.plot(results_D2.fittedvalues,color = 'red')
    plt.title('model_D2')
    plt.subplot(3,1,3)
    plt.plot(D1,color = 'blue')
    plt.plot(results_D1.fittedvalues,color = 'red')
    plt.title('model_D1')
    plt.show()
    #再次分解后进行预测
    A2_all,D2_all,D1_all = pywt.wavedec(data[:-1],'db4',mode='sym',level=2)
    pA2 = model_A2.predict(params = results_A2.params,start = 1,end = len(A2_all))
    pD2 = model_D2.predict(params = results_D2.params,start = 1,end = len(D2_all))
    pD1 = model_D1.predict(params = results_D1.params,start = 1,end = len(D1_all))
    denoised_index = pywt.waverec([pA2,pD2,pD1],'db4')
    denoised_index = denoised_index.reshape(-1,1)
    denoised_index = xsc.inverse_transform(denoised_index)
    denoised_index = denoised_index.reshape(-1)
    plt.figure()
    plt.plot(data[:-1],color = 'blue')
    plt.plot(denoised_index,color = 'red')
    plt.show()
    plt.figure()
    plt.plot(data[split_num+1:],color = 'blue')
    plt.plot(denoised_index[split_num+1:],color = 'red')
    plt.show()

# 判定结果：末端10%的预测值的MSE
i = 0
predictAns = svrPredict(stdData[i])
print(mean_squared_error(predictAns,stdData[i,-47:]))
