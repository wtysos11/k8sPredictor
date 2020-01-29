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
            print(lastkey)
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

# 第三部分，聚类