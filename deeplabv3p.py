import requests
import json
import cv2
import base64
import os
import numpy as np

# 图像分割


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


# 发送HTTP请求
data = {'images':[cv2_to_base64(cv2.imread("./img/girl.jpg"))]}
headers = {"Content-type": "application/json"}
url = "http://10.1.12.85:8866/predict/deeplabv3p_xception65_humanseg"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

result = r.json()["results"][0]['data']

with open(
        os.path.join("output", "out.jpg"),
        "wb") as fp:
    fp.write(base64.b64decode(result))

# 打印预测结果
# print(base64_to_cv2(result))
