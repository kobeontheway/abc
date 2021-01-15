# coding: utf8
import requests
import json
# 中文词法分析-done

if __name__ == "__main__":
    # 指定用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
    text = ["今天是个好日子", "天气预报说今天要下雨", "今天周五又去便利蜂采购了，本来是想去买蛋糕的，一回头发现有海带结！"]
    # 设置运行配置
    # 对应本地预测lac.cut(text=text, batch_size=1)
    data = {"text": text, "batch_size": 1}

    # 指定预测方法为lac并发送post请求，content-type类型应指定json方式
    # HOST_IP为服务器IP
    url = "http://10.1.12.33:8866/predict/lac"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
