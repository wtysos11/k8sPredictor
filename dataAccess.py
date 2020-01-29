# By using this script, it can download all the data from https://dumps.wikimedia.org/other/pageviews/ in the way I want.
# I plan to download all the data first, then unzip and aggregate them.
# 比较好的消息是找到了维基给的官方数据，虽然不够新，但也够用了：https://dumps.wikimedia.org/other/pagecounts-ez/
# 官方的数据不符合我的要求，但是我依据官方给的工具大概看了一下数据，都还是比较有周期性特性的，且比较相近的数据之间相似度很明显。只是不知道和全体相似度有没有关系
import os
import urllib
import urllib.request
import time
storePath = 'E:/code/myPaper/data' #文件存储路径
#格式化int型，使其为二位字符
def format2(ele):
    s = str(ele)
    if len(s)<2:
        s = '0'+s
    return s

#年月日，格式不要求对对齐
def downloadFileWithTime(year,month,day,hour):
    month = format2(month)
    day = format2(day)
    hour = format2(hour)
    print("downloading file at {}/{}/{}:{}".format(year,month,day,hour))
    site = {"year":year,"month":month,'day':day,'hour':hour}
    url = "https://dumps.wikimedia.org/other/pageviews/{year}/{year}-{month}/pageviews-{year}{month}{day}-{hour}0000.gz".format(**site)  
    print(url)
    LocalPath = os.path.join(storePath,'pageviews-{year}{month}{day}-{hour}0000.gz'.format(**site))
    begin = time.time()
    urllib.request.urlretrieve(url,LocalPath)
    end = time.time()
    print("finish {}s",end-begin)
    time.sleep(5)#休5s，防止被ban ip

year = 2019
month = 1
for day in range(1,32):
    for hour in range(24):
        downloadFileWithTime(year,month,day,hour)