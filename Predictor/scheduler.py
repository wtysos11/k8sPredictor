# 调度器，通过HTTP访问华为云的API以及prometheus拿到数据
import requests
import json
import math
token = ""
def validateToken():
    url = "https://iam.cn-south-1.myhuaweicloud.com/v3/auth/tokens"
    headers = {'X-Subject-Token':token,'X-Auth-Token':token}
    r = requests.get(url,headers = headers)
    if r.status_code != 200:
        return False
    else:
        return True
    
# 拿去token，成功返回token，失败返回空字符串
def getToken():
    url = 'https://iam.cn-south-1.myhuaweicloud.com/v3/auth/tokens'
    jsonStr = '''
    {
    	"auth": {
    	  "identity": {
    		"methods": ["password"],
    		"password": {
    		  "user": {
    			"name": "pmlpml0928",
    			"password": "yuy3226",
    			"domain": {
    			  "name": "pmlpml0928"
    			}
    		  }
    		}
    	  },
    	  "scope": {
    		"project": {
    		  "name": "cn-south-1"
    		}
    	  }
    	}
      }
    '''
    headers = {'Content-Type':'application/json'}
    try:
        resposne = requests.post(url=url,headers=headers,data=bytes(jsonStr,encoding='utf8'))
        token = resposne.headers['x-subject-token']
        return token
    except:
        return ""

# 拿到现有容器数
def getContainerNum():
    data = getContainer()
    return data['spec']['replicas']

# 拿到现有信息（JSON格式）
def getContainer():
    url = "https://95110d61-465a-11e9-856e-0255ac102279.cce.cn-south-1.myhuaweicloud.com/apis/extensions/v1beta1/namespaces/wtytest/deployments/cproductpage/scale"
    if not validateToken():
        token = getToken()
    if len(token) == 0:
        print('Error when getting token')
        return 0
    headers = {'X-Auth-Token':token,'Content-Type':"application/json"}
    response = requests.get(url=url,headers=headers)
    data = json.loads(response.text)
    return data
# 调整到现有容器数
def changeContainerNum(containerNum):
    try:
        curData = getContainer()
        url = "https://95110d61-465a-11e9-856e-0255ac102279.cce.cn-south-1.myhuaweicloud.com/apis/extensions/v1beta1/namespaces/wtytest/deployments/cproductpage/scale"
        if not validateToken():
            token = getToken()
        if len(token) == 0:
            print('Error when getting token')
            return 0
        headers = {'X-Auth-Token':token,'Content-Type':"application/json"}
        curData['spec']['replicas'] = containerNum
        jsData = json.dumps(curData)
        response = requests.put(url=url,headers=headers,data = jsData)
        if response.status_code != 200:
            return False
        return True
    except:
        return False
# 获取响应时间
def getResponseTime():
    url = 'http://139.9.57.167:9090/api/v1/query?query=sum(delta(istio_request_duration_seconds_sum{destination_workload_namespace=%27wtytest%27,reporter=%27destination%27,destination_workload=%27cproductpage%27}[15s]))/sum(delta(istio_request_duration_seconds_count{destination_workload_namespace=%27wtytest%27,reporter=%27destination%27,destination_workload=%27cproductpage%27}[15s]))%20*%201000'
    response = requests.get(url=url)
    data = json.loads(response.text)
    if len(data['data']['result'])==0:
        return 0
    else:
        res = data['data']['result'][0]['value'][1]
        if res == 'NaN':
            return 0
        else:
            return float(res)

# 获取当前流量
def getTraffic():
    url = "http://139.9.57.167:9090/api/v1/query?query=sum(rate(istio_requests_total{destination_workload_namespace='wtytest',reporter='destination',destination_workload='cproductpage'}[30s]))"
    response = requests.get(url=url)
    data = json.loads(response.text)
    if len(data['data']['result'])==0:
        return 0
    else:
        return float(data['data']['result'][0]['value'][1])

