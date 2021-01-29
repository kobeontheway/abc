import requests
import json

import soundfile as sf

# 发送HTTP请求

data = {'texts':['Simple as this proposition is, it is necessary to be stated',
                 'Parakeet stands for Paddle PARAllel text-to-speech toolkit'],
        'use_gpu':False}
headers = {"Content-type": "application/json"}
url = "http://10.1.12.85:8866/predict/transformer_tts_ljspeech"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 保存结果
result = r.json()["results"]
wavs = result["wavs"]
sample_rate = result["sample_rate"]
for index, wav in enumerate(wavs):
    sf.write(f"{index}.wav", wav, sample_rate)
