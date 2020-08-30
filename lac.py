import requests
import json

text = ["今天是个好日子", "天气预报说今天要下雨"]
data = {"texts": text, "batch_size": 1}
url = "http://10.1.12.64:8866/predict/lac"
headers = {"Content-Type": "application/json"}

r = requests.post(url=url, headers=headers, data=json.dumps(data))
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
