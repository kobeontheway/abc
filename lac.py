# coding: utf8
import requests
import json
# 中文词法分析-done

if __name__ == "__main__":
    # 指定用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
    text = ["今天是个好日子", "天气预报说今天要下雨"]
    # 以key的方式指定text传入预测方法的时的参数，此例中为"data"
    # 对应本地部署，则为lac.analysis_lexical(texts=[text1, text2])
    data = {"texts": text, "batch_size": 2}
    # 指定预测方法为lac并发送post请求
    url = "http://10.1.12.33:8866/predict/lac"
    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
