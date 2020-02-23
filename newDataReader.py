#新式文件读取器，参考dataReader.py
# 1. 先读取所有文件，求和，如果没有条目则删除，取最大的前1w个
# 2. 对前1w个的时序数据进行整理，
import os

def format2(ele):
    s = str(ele)
    if len(s)<2:
        s = '0'+s
    return s

#找到名字
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

# 首先构建一个足够大的存储结构，读取第一个文件建立目录并录入
# 使用library存储最终的数据
reader = open(LocalPath,'r',encoding='utf-8')
store = reader.readlines()#临时存储，必须删除
reader.close()
def Add2Lib(key,value):
    if key not in library:
        library[key]=value
    else:
        library[key]+=value    

#将所有的英文数据加入到存储中，构建名字
for line in store:
    elements = line.split(' ')
    if elements[0]=='en':
        Add2Lib(elements[1],0)  
        
del store
# 访问所有的数据
# 对于后面的所有访问记录，将数值填充进目录之中。如果不存在该条目，直接废弃

# 将指定时间的数据添加到指定的条目中
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
    # 在遍历的过程中进行检查，如果有键没有在该文件中出现，直接删除
    rest = dict.fromkeys(library.keys(),0)
    for line in store:
        elements = line.split(' ')
        if elements[0]=='en':
            if elements[1] in library:
                library[elements[1]] += int(elements[2])
                del rest[elements[1]]
    
    # 对于rest中还剩下的，即为需要被删除的
    for k in rest.keys():
        del library[k]
    del store
# 找到最大的1w个
# 重复访问所有文件，构建时序数据库。
for day in range(1,21):
    for hour in range(24):
        add2Data(year,month,day,hour)
        print(year,month,day,hour)

# 进行排序，找到前1w个
num = list(library.values())
num.sort()
divNum = num[-10000]

#将所有比这个值要小的删除
keys = list(library.keys())
for key in keys:
    if library[key]<divNum:
        del library[key]

##########################
# 再次访问数据，读入时序数据库
data = {}
for name in library.keys():
    data[name]=[]

def add2DataList(year,month,day,hour):
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


# 将数据写入到文件中
import os
fileName = 'result.txt'
dataPath = "E:\\code\\myPaper\\k8sPredictor"
def WriteDataToFile(storeData):
    LocalPath = os.path.join(dataPath,fileName)
    writer = open(LocalPath,'w',encoding='utf-8')
    for key in storeData:
        writer.write('keykey\n')
        writer.write(key+'\n')
        for ele in storeData[key]:
            writer.write(str(ele)+'\n')
    writer.close()

WriteDataToFile(library)