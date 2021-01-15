import requests
import json
# 文本相似度

# 待预测数据
test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]

text = [test_text_1, test_text_2]

# 设置运行配置
# 对应本地预测simnet_bow.similarity(texts=text, batch_size=1, use_gpu=True)
data = {"texts": text, "batch_size": 1, "use_gpu":False}

# 指定预测方法为simnet_bow并发送post请求，content-type类型应指定json方式
# HOST_IP为服务器IP
url = "http://10.1.12.33:8866/predict/simnet_bow"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))

# 命令行模式：
# hub run simnet_bow --text_1 "这道题太难了" --text_2 "这道题不简单"
