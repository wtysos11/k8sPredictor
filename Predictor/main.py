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
    fileName = 'trafficTotal.txt'
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

#根据容器所能正常运行的上限容量来进行调度
def getConNumFromDict(futureTraffic,trafficDict,delta = 10):
    for i in range(2,7):
        if trafficDict[i]>=futureTraffic+delta:
            return i
    return 7

#阈值法调度策略：如果响应时间超过100，则增加。反之如果响应时间小于60则减少
def reactiveScheduler(rspTime,containerNum):
    upThre = 60
    downThre = 40
    if rspTime>upThre:
        containerNum += 1
    elif rspTime<downThre:
        containerNum -= 1

    if containerNum<2:
        containerNum = 2
    elif containerNum > 7:
        containerNum = 7
    return containerNum

# 预测维持响应时间在100ms以下
# 2:133,3:275,4:360,5:430,6:500
def getContainerFromTraffic(real,pred,responseTime,containerNum,cnList):
    trafficThreshold = {2:150,3:240,4:330,5:410,6:480}
    upThre = 60
    downThre = 30
    #如果预测的流量过小，直接转为阈值法
    if len(real)<2: #开始几个点保持不变
        return containerNum
    # 如果预测的流量不足三个容器，直接忽视
    if real[-1]<100:
        print('0:数量过小，使用阈值法')
        return reactiveScheduler(responseTime,containerNum)

    # 如果当前已经超时，则直接进行阈值法调度
    if responseTime > upThre:
        print('1：当前已经超时，阈值法调度')
        containerNum += 1
        return containerNum

    # 如果当前的预测值与真实值很接近，则直接按照预测值预测
    # 如果当前的预测值与真实值不接近，但是斜率接近，则按照预测值预测斜率
    # 如果都不接近，则转为阈值法调度器，直接按照响应时间预测。对于超过200的响应时间直接+1
    AggresiveDelta = 50
    ConserveDelta = 30
    if len(pred)>=2:
        p = getConNumFromDict(pred[-1],trafficThreshold)
        if p < 2:
            p = 2
        if p>containerNum and abs(real[-1]-pred[-2])<AggresiveDelta:
            #激进的增加策略，允许一定误差的增加
            print('2:预测准确，激进增加，直接使用预测')
            return p
        elif abs(real[-1]-pred[-2])<ConserveDelta:
            #减少容器数必须要数次预测正确才能够进行
            print('2:预测准确，保守减少，直接使用预测')
            # 如果当前p值需要减少
            if (p < containerNum and real[-1]<real[-2] and real[-2]<real[-3] and abs(real[-2]-pred[-3])<ConserveDelta) or rspTime < downThre:
                #检验之前需要持续下降，且两个点都预测准确，
                p = containerNum - 1
                return p
            else:
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
            print('3:存在相关性：',v,p)
            r1 = real[-1]-real[-2]
            r2 = real[-2]-real[-3]
            p1 = pred[-2]-pred[-3]
            p2 = pred[-3]-pred[-4]
            #比例计算
            if p1==0:
                ratio1 = 0
            else:
                ratio1 = r1/p1
            if p2==0:
                ratio2 = 0
            else:
                ratio2 = r2/p2
            if abs(ratio1 - ratio2)<1: 
                #预测的趋势很相近，可以直接沿用。具体表现为比例相近
                print('3.1:趋势接近，',ratio1,ratio2)
                futureTraffic = real[-1] + (pred[-1]-pred[-2])
                p = getConNumFromDict(futureTraffic,trafficThreshold)
                print('futureTraffic:',futureTraffic,p)
                # 观察趋势本身，如果是增加趋势，允许增加。
                # 如果是减少趋势，允许减少
                if p<2:
                    p=2
                #必须是同趋势才能够成立
                if p < containerNum and pred[-1] < pred[-2]:
                    p = containerNum - 1
                    return p
                elif p>containerNum and pred[-1] > pred[-2]:
                    return p
            else:
                #尽管比例不相近，但是趋势相近，可以进行试探。即如果一直增加，则增加的范围会扩大。
                # 如果流量一直降低，则对下降的流量进行一定的容纳
                print('3.2:趋势不接近，',ratio1,ratio2)
                if r1>0 and r2>0 and p1>0 and p2>0 and ratio1!=0:
                    #一直在增长，增加容纳的判断
                    futureTraffic = real[-1] + (pred[-1]-pred[-2])*ratio1
                    print('始终在增加')
                    p = getConNumFromDict(futureTraffic,trafficThreshold,50)
                    if p<2:
                        p = 2
                    if p > containerNum: #增加趋势，只能够调整增加
                        return p
                elif r1<0 and r2<0 and p1<0 and p2<0 and pred[-1]<pred[-2]:
                    # 一直在减少，且响应时间小于一个非常小的值
                    if responseTime < downThre:
                        print('长期减少，进行调度')
                        p -= 1
                        if p<2:
                            p=2
                        return p
                    else:
                        #判断一段时间内都在减少，并且过去一段时间内容器数量不能增加
                        if real[-3] < real[-4] and cnList[-1]<=cnList[-2] and cnList[-2]<=cnList[-3]:
                            futureTraffic = real[-1] + (pred[-1]-pred[-2])*ratio1
                            p = getConNumFromDict(futureTraffic,trafficThreshold,50)
                            if p<2:
                                p = 2
                            print('一段时间减少',futureTraffic,p)
                            return p

    # 阈值法只根据当前是否超时/过低来判断是否需要增减
    print('eternal:达到底部')
    return reactiveScheduler(responseTime,containerNum)

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
    changeContainerNum(5)
    
    start_time = "2020-03-08 09:55:00"
    start = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    isOk = False
    while not isOk:
        current = datetime.datetime.now()
        if current.hour >= start.hour and current.minute>=start.minute and current.second>=start.second:
            isOk = True
        else:
            time.sleep(1)
    
    real = [] #预测的第一个点为真实流量
    time.sleep(5) #先延迟5s，避免开场触发调度

    i=0#迭代开始
    print('enter')
    #提前调度容器数量为7
    cnList = []
    while(i<=47): # 进行47次调度与记录
        # 测试数据为48分钟，测试点共48个点
        # 第一个点统一为真实值，每次调度会提前30s进行
        # 即每次30的时候进行调度，00的时候获取流量

        currentTime = datetime.datetime.now()
        if currentTime.second - 0>=0 and currentTime.second - 0<=2: #如果在30s内
            # 进行前置调度
            # 调用函数，根据预测值拿到
            if i==0:#跳过第一次调度
                continue
            print(real[:i],pred[:i],pred[i])
            rspTime = getResponseTime()
            conNum = getContainerNum()
            cnList.append(conNum)
            containerNum = getContainerFromTraffic(real[:i],pred[:i+1],rspTime,conNum,cnList)
            #containerNum = reactiveScheduler(rspTime,conNum)
            changeContainerNum(containerNum)
            print('schedule',i,' change:',containerNum-conNum,'rspTime:',rspTime,'conNum',containerNum)
            time.sleep(5)
        elif currentTime.second - 30>=0 and currentTime.second - 30<=2: #如果在开始的第一秒内
            # 记录下当前的流量值，写入列表中
            i+=1
            traffic = int(getTraffic())
            print('time '+str(i)+'traffic:',traffic)
            #紧急的阈值法调度，如果有超时情况马上进行调度

            real.append(traffic)
            time.sleep(5)
        else:
            time.sleep(1)

