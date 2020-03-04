# 主函数部分
# 通过scheduler.py来实现对各项指标的获取与容器的调度
# 通过提前运行相关的聚类算法和调度工具来进行选择

import datetime
import time
import os
from scheduler import * 
from scipy.stats import spearmanr
import numpy as np

# 读取流量数据
def readTrafficFromFile():
    fileName = 'traffic.txt'
    dataPath = "E:\\code\\myPaper\\k8sPredictor"
    LocalPath = os.path.join(dataPath,fileName)
    reader = open(LocalPath,'r',encoding='utf-8')
    store = reader.readlines()
    reader.close()
    data = []
    for line in store:
        data.append(int(float(line[:-1])))
    del store
    return data

def getContainerFromTraffic(real,pred,responseTime,containerNum):
    #如果预测的流量过小，直接转为阈值法
    if len(real)<1: #初始的时候保持不变
        return containerNum
    # 如果预测的流量不足三个容器，直接忽视
    if real[-1]<200:
        print('0:数量过小，使用阈值法')
        if responseTime > 250:
            containerNum += 1
        elif responseTime < 50:
            containerNum -= 1
        if containerNum < 1:
            containerNum = 1
        return containerNum

    # 如果当前的预测值与真实值很接近，则直接按照预测值预测
    # 如果当前的预测值与真实值不接近，但是斜率接近，则按照预测值预测斜率
    # 如果都不接近，则转为阈值法调度器，直接按照响应时间预测。对于超过200的响应时间直接+1
    delta = 50
    if len(pred)>=2:
        print('1:预测准确，直接使用预测')
        if abs(real[-1]-pred[-2])<delta:#如果在范围以内。可能性很小，适用于拟合的比较好的情况（比如单个的预测）
            p = math.ceil(pred[-1]/80)
            if p < 1:
                p = 1
            return p

    # 如果当前已经超时，则直接进行阈值法调度
    if responseTime > 250:
        print('2：当前已经超时，阈值法调度')
        containerNum += 1
        return containerNum
    elif responseTime < 50:
        print('2：当前时间过少，阈值法调度')
        containerNum -= 1
        return containerNum
    if containerNum < 1:
        print('2:当前流量过小，已经触底')
        containerNum = 1
        return containerNum
    

    # 使用spearman进行趋势判断
    # 对于v>0.6且p<0.2的结果，认为是同趋势
    # 对于同趋势的结果，分两种情况。分别比较最近的三个点的两个差值，如果都相同，那就直接预测
    # 如果不相同且整体相差一个比例，那就按照比例预测
    # 如果不相同且不按照比例，那就加权预测

    # 如果长度不够5，则直接忽略
    if len(real)>=5:
        v,p = spearmanr(real[-5:],pred[-6:-1])
        if v>0.6 and p<0.2:
            print('3:存在相关性')
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
                print('3.1:趋势接近')
                futureTraffic = real[-1] + (pred[-1]-pred[-2])
                p = math.ceil(futureTraffic/80)
                if p<1:
                    p=1
                return p
            elif abs(ratio1)<1 or (sum(delta>2)<2):# 尽管比例不相近，但是整体趋势接近。预测方式是综合判断，用过去两个点的值来预测下一个点
                #判断比例
                # 使用实际值与预测值的插值比例来进行预测
                print('3.2:趋势不接近，但是比例接近')
                avgDiff1 = 0.5*r1+0.5*r2
                avgDiff2 = 0.5*p1+0.5*p2
                newDiff = 0.6*(pred[-1]-pred[-2])+0.3*p1+0.1*p2
                futureTraffic = real[-1] + (newDiff/avgDiff2*avgDiff1)# 说实话，效果差强人意
                p = math.ceil(futureTraffic/80)
                if p<1:
                    p=1
                return p
    # 阈值法只根据当前是否超时/过低来判断是否需要增减
    print('eternal:达到底部')
    if responseTime > 250:
        containerNum += 1
    elif responseTime < 50:
        containerNum -= 1
    if containerNum < 1:
        containerNum = 1
    return containerNum

if __name__=="__main__":
    # 事先应该有一个dataGenerator.py的文件，将指定的流量进行聚类后的预测，完全随机
    # 本文件直接读取预测的流量，预测流量长度应该为47，加上原有的433个真实数据拼接而成。第一个点的值直接使用真实值进行
    # 本文件应该与流量测试工具同时运行

    # 进行线性映射，就测试的结果而言，单个容器承受的流量在80req/s会比较高，90req/s就会超时

    # 步骤
    # 启动时间可以考虑使用定时器完成，实现完全同步
    # 1. 读取已经完成的预测流量
    # 2. 每个时刻，获取现在的流量，结合判断下一个点的调度（调度需要提前进行）
    # 3. 进行调度。重复2

    # 提前读取文件
    pred = readTrafficFromFile() #应该有48个点，第一个点为真实流量，后面的为预测值
    changeContainerNum(7)
    
    start_time = "2020-03-02 15:14:00"
    start = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    isOk = False
    while not isOk:
        current = datetime.datetime.now()
        if current.hour >= start.hour and current.minute>=start.minute and current.second>=start.second:
            isOk = True
        else:
            time.sleep(1)
    
    real = [pred[0]] #预测的第一个点为真实流量
    time.sleep(5) #先延迟5s，避免开场触发调度
    # 装载真实值
    # 到达指定时间
    i=0
    print('enter')
    #提前调度容器数量为7
    
    while(i<48): # 进行47次调度与记录
        # 测试数据为48分钟，测试点共48个点
        # 第一个点统一为真实值，每次调度会提前30s进行
        # 即每次30的时候进行调度，00的时候获取流量
        currentTime = datetime.datetime.now()
        if currentTime.second - 30>=0 and currentTime.second - 30<=1: #如果在30s内
            # 进行前置调度
            # 调用函数，根据预测值拿到
            print(real,pred)
            rspTime = getResponseTime()
            conNum = getContainerNum()
            if i>0:
                containerNum = getContainerFromTraffic(real[:i-1],pred[:i],rspTime,conNum)
            else:# 第一次不进行调度
                containerNum = conNum
            changeContainerNum(containerNum)
            print('schedule',i,' change:',containerNum-conNum)
            time.sleep(5)
            i += 1
        elif currentTime.second - 0>=0 and currentTime.second - 0<=1: #如果在开始的第一秒内
            # 记录下当前的流量值，写入列表中
            traffic = int(getTraffic())
            print('traffic:',traffic)
            real.append(traffic)
            time.sleep(5)
        else:
            time.sleep(1)

#欠缺的问题：如何根据实时记录的值来估算出容器数量。