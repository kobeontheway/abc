# coding: utf8
import requests
import json
# 中文情感分析-done

if __name__ == "__main__":
    # 指定用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
    text_list = ["我不爱吃甜食", "我喜欢躺在床上看电影"]
    data = {"text": text_list}
    # 指定预测方法为senta_lstm并发送post请求
    url = "http://10.1.12.33:8866/predict/text/senta_lstm"

    r = requests.post(url=url,  data=data)

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
