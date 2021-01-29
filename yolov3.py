import requests
import json
import cv2
import base64
# 目标检测

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

# 发送HTTP请求
data = {'images':[cv2_to_base64(cv2.imread("./img/dog.jpg"))]}
headers = {"Content-type": "application/json"}
url = "http://10.1.12.85:8866/predict/yolov3_darknet53_coco2017"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
