from dataOperator import DataOperator


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA,ARMA
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
import time
import numpy as np
import time
##评价系统
#根据真实值评价预测器的预测效果，使用MSE
# data为测试集
def evaluatePrediction(data,pred):
    score = 0
    for i in range(len(data)):
        score+= mean_squared_error(data[i,-47:,0],pred[i])
    score/=len(data)
    return score

##########################
# ARMA预测，调用getARMAresult获得结果，ratio为预测集的比例
def armaPredict(data,ratio):
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
            result.append(pre)
        print(len(result))
    e = time.time()
    print('arma predict time:',e-s,'s')
    return result

# 导入所有数据，返回一个n*47的预测结果
def getARMAresult(cache):
    data = cache.stdData
    armaResult = armaPredict(data,0.9)
    return armaResult

##########################
#####SVR调用
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

def svrPredict(data):
    s = time.time()
    store = []
    for i in range(len(data)):
        y_test,y_prediction = getPredictResultWithSlidingWindows(data[i])
        store.append(y_prediction)
    e = time.time()
    print('svr predict time:',e-s,'s')
    return store


##RNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
import time
def RNNPredict(data):
    ratio = 0.9
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
    return store

# LSTM设计
# 基础想法是根据过去1天的数据来预测未来1天的某个小时的数据。计划是输入24个小时，输出未来24个小时中的某个小时
# 输入特征：24小时的时间戳，共24个统计量
# 均值、方差、一阶差分 25个统计量
# 预测的数据是1天中的第几个小时

# 预想的架构：首先是24个时间戳，24*1，LSTM扩展到24*3，然后Dropout，再进行LSTM缩减到平的18，加上外部特征整合后接三层Dense
def LSTMPredict(data):
    ratio = 0.9
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
    return store

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
def getLSTMResult(data,epch):
    #第一步，切分数据集
    ratio = 0.9
    #然后对于每个24小时进行训练。用前24小时去精确预测后24小时中的某一个点。
    # 对训练数据的24小时内生成训练窗口集合。
    # 内部时间戳，24个连续的时间戳
    # 外部辅助特征，23个差分+平均值+方差+1天中的第几个小时
    s = time.time()
    store = []
    for index in range(len(data)):
        # 对于每一个时间序列数据
        main_input_list = []
        auxiliary_input_list = []
        y_train = []
        # 对于每一个点，将其前面的24个点打包成连续时间戳
        # 提取前24个点的均值和方差，以及23个一阶差分数据
        # 以及这个点是该天的第几个小时
        window_size = 24
        tsData = data[index]
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
        store.append(y_test)
        del model
    e = time.time()
    print('lstm predict time:',e-s,'s')
    return store



if __name__=="__main__":
    cache = DataOperator()
    epch = [14,24]
    l = []
    # 问题：LSTM可能与投入的数据量有关系
    for e in epch:
        data = cache.stdData[:1000]
        ans = getLSTMResult(data,e)
        l.append(evaluatePrediction(data,ans))
    for i in range(len(l)):
        print(i,epch[i],l[i])
    # 检测ARMA
    #ans = getARMAresult(cache)
    #print(evaluatePrediction(cache.stdData,ans))
    # 检测SVR
    #svr = svrPredict(cache)
    #print(evaluatePrediction(cache.stdData,svr))

