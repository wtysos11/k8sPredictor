'''
最终目标：得到若干类，每一类的流量数据满足同增同减的性质，可以通过预测一个中心流量并进行某种线性变换得以预测其他的流量。
文件目的：
1. 读取文件
2. 选出所有en中最大的1w个数据
3. 将其转换成时序格式（如果有漏的，直接删除）
4. 对结果分别进行DBSCAN聚类、谱聚类等
'''

# 读取文件并筛选
# 首先读取一个文件，选出en中最大的前1w个数据
# 然后记录下一张表格，以后每个文件中只选择这些数据进行添加。对于没有的数据直接删除整个项
import os

def format2(ele):
    s = str(ele)
    if len(s)<2:
        s = '0'+s
    return s

library = {}#以地址为key的词典

storePath = "E:\\code\\myPaper\\data"
year = 2019
month = 1
day = 1
hour = 0
month = format2(month)
day = format2(day)
hour = format2(hour)
site = {"year":year,"month":month,'day':day,'hour':hour}
LocalPath = os.path.join(storePath,'pageviews-{year}{month}{day}-{hour}0000'.format(**site))

reader = open(LocalPath,'r',encoding='utf-8')
store = reader.readlines()#临时存储，必须删除
reader.close()
def Add2Lib(key,value):
    if key not in library:
        library[key]=value
    else:
        library[key]+=value    

#将所有的英文数据加入到存储中
for line in store:
    elements = line.split(' ')
    if elements[0]=='en':
        Add2Lib(elements[1],int(elements[2]))  
        

del store

# 统计出最大的k个数字，将其key提取出来
# 首先对元组按照第二项进行逆序排序
kNum = 10000 #取前1w个
result = sorted(list(library.items()),key=lambda x:x[1],reverse=True)
result = result[:kNum]
del library

# 统计名字，将名字导入字典中形成集合，记录下当前数据
data = {}
for name,num in result:
    data[name]=[num]


#将指定文件的数据顺序添加到data中，要求data中已有指定条目
def add2Data(year,month,day,hour):
    y = year
    m = format2(month)
    d = format2(day)
    h = format2(hour)
    site = {"year":y,"month":m,'day':d,'hour':h}
    LocalPath = os.path.join(storePath,'pageviews-{year}{month}{day}-{hour}0000'.format(**site))
    reader = open(LocalPath,'r',encoding='utf-8')
    store = reader.readlines()#临时存储，必须删除
    reader.close()
    #遍历，将有的添加入data中
    for line in store:
        elements = line.split(' ')
        if elements[0]=='en':
            if elements[1] in data:
                data[elements[1]].append(int(elements[2]))
    
    del store
# 对其他文件进行读取，只用指定的数字
# 最后统计，对于长度不对的项目进行删除
for day in range(1,32):
    for hour in range(24):
        if day==1 and hour ==0:
            continue
        add2Data(year,month,day,hour)
        print(year,month,day,hour)



# 第二部分，数据处理
#进行资格筛查，如果长度不够，说明有缺漏
correctNum = 480
for key in list(data.keys()):
    if len(data[key])!=correctNum:
        del data[key]

# 第二部分-1 文件存储
# 进行数据备份与数据读取
# 文件规定：以keykey开头下一行为键名，然后为一串int列表
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

# 将原始数据复制，然后归一化处理
# 去除outlier，先排序，然后对上下数量分别为1%的点进行最大/最小值抑制
# 第三部分，聚类

# 对于给定的列表，抑制前ratio%和后ratio%的数据
# data会直接修改
#def suppressMinAndMax(data,ratio):
#    anotherdata = sorted(data)#顺序排序
#    num = len(anotherdata)
#    minVal = anotherdata[int(num*ratio)]
#    maxVal = anotherdata[-1*int(num*ratio)]
#    for i in range(len(data)):
#        if data[i]<minVal:
#           data[i] = minVal
#        elif data[i]>maxVal:
#           data[i] = maxVal
#
    
#转成tslearn格式
import time
data = ReadDataFromFile()

from tslearn.utils import to_time_series_dataset
formatted_dataset = to_time_series_dataset(list(data.values()))

#归一化
from tslearn.preprocessing import TimeSeriesScalerMinMax
scaler =  TimeSeriesScalerMinMax(value_range=(0., 1.))
storeMinMax = []
for i in range(len(formatted_dataset)):
    ele = formatted_dataset[i]
    ele = ele.reshape(ele.shape[0])
    storeMinMax.append((min(ele),max(ele)))
    formatted_dataset[i] = scaler.fit_transform(ele).reshape(formatted_dataset[i].shape[0],formatted_dataset[i].shape[1])

#进行聚类

## 目标：将已有的时序数据分成若干类，每个类中有一个中心变量，通过对中心变量的预测可以实现对其他时序数据的预测，从而降低时序数据预测的开销
## 暂时不考虑时序上的位移，只考虑数值上每个类与该类之间存在固定的线性变化，这样通过计算线性关系的算法可以直接算出来
# T1：直接对原始数据进行基于切分的聚类时不可行的，速度太慢。


# 想法一：进行普通归一化。然后以欧式距离/相关性进行聚类，聚类方法为普通K-means递进/DBSCAN/层次聚类

# 想法二：进行最大最小值抑制的归一化，去除噪点，然后同上。
# 异常点数量定为前后1%，先排序，然后设置阈值，再进行排除
# 抑制最小最大值的归一化

#进行最小最大值抑制
import numpy as np
ratio = 0.01#抑制前1%与后1%的值
stdData = formatted_dataset.copy()
for i in range(len(stdData)):
    ele = stdData[i].copy()
    ele = ele.reshape(ele.shape[0])
    ele.sort()
    minNum,maxNum = ele[int(ratio*len(ele))],ele[-1*int(ratio*len(ele))]
    del ele
    def suppressMinandMax(ele):
        if ele<minNum:
            return minNum
        elif ele>maxNum:
            return maxNum
        else:
            return ele
    cache = stdData[i]
    stdData[i] = np.fromiter(map(suppressMinandMax,cache.reshape(cache.shape[0])),dtype = np.float64).reshape(cache.shape)

#归一化
from tslearn.preprocessing import TimeSeriesScalerMinMax
scaler =  TimeSeriesScalerMinMax(value_range=(0., 1.))
storeMinMax = []
for i in range(len(stdData)):
    ele = stdData[i]
    ele = ele.reshape(ele.shape[0])
    storeMinMax.append((min(ele),max(ele)))
    stdData[i] = scaler.fit_transform(ele).reshape(stdData[i].shape)

# 想法三：进行归一化，然后做FFT或小波变化，然后再进行聚类（可以用这个来验证聚类算法）

# 对一个numpy数组进行DCT
from numpy import fft


# 想法四：利用PAA等技术
# 然后使用sklearn等库
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation
import time
# 1dSAX
n_paa_segments = 40
n_sax_symbols_avg = 30
n_sax_symbols_slope = 30
one_d_sax = OneD_SymbolicAggregateApproximation(
    n_segments=n_paa_segments,
    alphabet_size_avg=n_sax_symbols_avg,
    alphabet_size_slope=n_sax_symbols_slope)
transformed_data = one_d_sax.inverse_transform(one_d_sax.fit_transform(stdData))


from sklearn.cluster import MiniBatchKMeans,KMeans,DBSCAN,SpectralClustering,Birch
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score

n_cluster = 100

#Kmeans 结果
# 超参数：k的取值
s = time.time()
km = KMeans(n_clusters = n_cluster,random_state = 0)
y_pre = km.fit_predict(transformed_data)
e = time.time()
print(e-s,"s")
print(davies_bouldin_score(transformed_data,y_pre))

# MiniBatch k means
# 超参数：k的取值，batch_size的大小。其中batch_size的大小影响速度
s = time.time()
km = MiniBatchKMeans(n_clusters = n_cluster,init_size = 3*n_cluster,random_state = 0,batch_size = 10)
y_pre = km.fit_predict(transformed_data)
e = time.time()
print(e-s,"s")
print(davies_bouldin_score(transformed_data,y_pre))

# Birch
# 超参数：threshold阈值，branching_factor
s = time.time()
y_pre = Birch(n_clusters=None).fit_predict(transformed_data)
e = time.time()
print(e-s,"s")
print(davies_bouldin_score(transformed_data,y_pre))


# DBSCAN
# 超参数：eps,min_samples
s = time.time()
y_pre = DBSCAN(eps = 0.5,min_samples=5).fit_predict(transformed_data)
e = time.time()
print(e-s,"s")
print(davies_bouldin_score(transformed_data,y_pre))

# 谱聚类
# 超参数：加入n_cluster，gamma
s = time.time()
y_pre = SpectralClustering().fit_predict(transformed_data)
e = time.time()
print(e-s,"s")
print(davies_bouldin_score(transformed_data,y_pre))

# 猜想6：新型归类方法
# 多次聚类方法
# 1. 先进行PAA+平均值归一化(SAX)
# 2. 第一层聚类，再次抽象成三值数组，进行快速聚类
# 3. 第二层聚类，对于同一类内再次进行聚类（可以使用kmeans）
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation,SymbolicAggregateApproximation
from tslearn.utils import to_time_series_dataset
import time
import numpy as np
# 对于给定的原始数据(n*m*特征值数量型)，返回平均值归一化后的结果（不在原数据上进行变动）
def getStdData(originData):
    n_paa_segments = 120 #一天分成4份，每6个小时整合为一段
    paa_data = PiecewiseAggregateApproximation(n_segments = n_paa_segments).fit_transform(originData)
    #进行平均值归一化
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    dataset = scaler.fit_transform(paa_data)
    dataset = dataset.reshape(dataset.shape[0],dataset.shape[1])
    return dataset

#将归一化后的数据变成0、1、-1的数据集合。
#每个元素的意义为下一个元素是否会增加或减少
def get01Data(originData):
    ratio = 0.1
    data = originData.copy()
    for index,ele in enumerate(data):
        #对每个元素，确定其最大最小的范围。
        dataRange = max(ele) - min(ele)
        dataDiff = np.diff(ele)
        for i,num in enumerate(dataDiff):
            if num > dataRange*ratio:
                data[index][i] = 1
            elif num < dataRange*-1*ratio:
                data[index][i] = -1
            else:
                data[index][i] = 0
        data[index][len(ele)-1]=0
    return data

#原始数据为originData，直接从文件中得到的

fileData = ReadDataFromFile()
originData = to_time_series_dataset(list(data.values()))
stdData = getStdData(originData)
norData = get01Data(stdData)

