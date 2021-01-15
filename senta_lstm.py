import requests
import json

# 中文情感分析

# 待预测数据
text = ["这家餐厅很好吃", "这部电影真的很差劲"]

# 设置运行配置
# 对应本地预测senta_lstm.sentiment_classify(texts=text, batch_size=1, use_gpu=True)
data = {"texts": text, "batch_size": 1, "use_gpu":True}

# 指定预测方法为senta_lstm并发送post请求，content-type类型应指定json方式
# HOST_IP为服务器IP
url = "http://10.1.12.33:8866/predict/senta_lstm"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
