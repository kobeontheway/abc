# coding: utf8
import requests
import json
# 敏感词检测

if __name__ == "__main__":
    # 指定用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
    text_list = ["黄片下载", "中国黄页", "江泽民"]
    text = {"text": text_list}
    # 指定预测方法为lac并发送post请求
    url = "http://10.1.12.33:8866/predict/text/porn_detection_lstm"
    r = requests.post(url=url, data=text)

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
