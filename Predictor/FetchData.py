import datetime
import time
import urllib.parse
import requests

throughput_api = "sum(rate(istio_requests_total{destination_workload_namespace='wtytest',reporter='destination',destination_workload='cproductpage'}[30s]))"
responseTime_api = "sum(delta(istio_request_duration_seconds_sum{destination_workload_namespace='wtytest',reporter='destination',destination_workload='cproductpage'}[15s]))/sum(delta(istio_request_duration_seconds_count{destination_workload_namespace='wtytest',reporter='destination',destination_workload='cproductpage'}[15s])) * 1000"
supplyPod_api = "count(sum(rate(container_cpu_usage_seconds_total{image!='',namespace='wtytest',pod_name=~'cproductpage.*'}[10s])) by (pod_name, namespace))"
demandPod_api = "instance_predict_v1"
requestSuccessTotal_api = "sum(istio_requests_total{destination_workload_namespace='wtytest',reporter='destination',destination_workload='cproductpage',response_code=~'2.*'})"
requestFailTotal_api = "sum(istio_requests_total{destination_workload_namespace='wtytest',reporter='destination',destination_workload='cproductpage',response_code=~'5.*'})"
serviceTimeFailTotal_api = "sum(istio_request_duration_seconds_sum{destination_workload_namespace='wtytest',reporter='destination',destination_workload='cproductpage',response_code=~'5.*'})"
serviceTimeAvaliableTotal_api = "sum(istio_request_duration_seconds_sum{destination_workload_namespace='wtytest',reporter='destination',destination_workload='cproductpage',response_code=~'2.*'})"
cpuAvailability_api = "(sum(sum(rate(container_cpu_usage_seconds_total{image!='',namespace='wtytest',container_name!='istio-proxy'}[30s])) by (pod_name, namespace)) / sum(container_spec_cpu_quota{image!='',namespace='wtytest',container_name!='istio-proxy'} / 100000)) * 100 * 3 / 4"

def fetch_data(api_str, start_time, latsted_time, filename):
    pout = open(filename, "w")
    start = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    encoded_api = urllib.parse.quote_plus(api_str)
    for i in range(0, latsted_time, 5):
        t = start + datetime.timedelta(0, i)
        unixtime = time.mktime(t.timetuple())
        api_url = "http://139.9.57.167:9090/api/v1/query?time={}&query={}".format(unixtime, encoded_api)
        print(api_url)
        res = requests.get(api_url).json()["data"]
        if "result" in res and len(res["result"]) > 0 and "value" in res["result"][0]:
            v = res["result"][0]["value"]
            sv = str(v[1])
            if sv == "NaN":
                print("0")
                print("0", file=pout)
            else:
                print(sv)
                print(sv, file=pout)
        else:
            print("0")
            print("0", file=pout)
    pout.close()

start_time = "2020-02-19 10:30:00"
lasted_time = 2880

fetch_data(throughput_api, start_time, lasted_time, "throughput.log")
fetch_data(responseTime_api, start_time, lasted_time, "responseTime.log")
fetch_data(supplyPod_api, start_time, lasted_time, "supplyPod.log")
fetch_data(cpuAvailability_api, start_time, lasted_time, "cpuAvailability.log")
#fetch_data(demandPod_api, start_time, lasted_time, "demandPod6.log")
fetch_data(requestSuccessTotal_api, start_time, lasted_time, "requestSuccessTotal.log")
fetch_data(requestFailTotal_api, start_time, lasted_time, "requestFailTotal.log")
fetch_data(serviceTimeFailTotal_api, start_time, lasted_time, "serviceTimeFailTotal.log")
fetch_data(serviceTimeAvaliableTotal_api, start_time, lasted_time, "serviceTimeAvaliableTotal.log")
