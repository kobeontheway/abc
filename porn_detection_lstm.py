import requests
import json

# 待预测数据
text = ["黄片下载", "打击黄牛党"]

# 设置运行配置
# 对应本地预测porn_detection_lstm.detection(texts=text, batch_size=1, use_gpu=True)
data = {"texts": text, "batch_size": 1, "use_gpu":False}

# 指定预测方法为porn_detection_lstm并发送post请求，content-type类型应指定json方式
# HOST_IP为服务器IP
url = "http://10.1.12.85:8866/predict/porn_detection_lstm"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
