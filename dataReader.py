'''
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