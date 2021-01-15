import json
import requests
# 图像分类

file_list = ["img/cat.jpg", "img/flower.jpg"]
files = [("image", (open(item, "rb"))) for item in file_list]
url = "http://10.1.12.33:8866/predict/image/vgg11_imagenet"
r = requests.post(url=url, files=files)

print(json.dumps(r.json(), indent=4, ensure_ascii=False))
