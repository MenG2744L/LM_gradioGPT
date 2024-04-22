import numpy as np
import requests

# 文字转语音
def run(output):
    response = requests.post(f"http://127.0.0.1:8000/infer/{output}")   # 使用requests库的post方法向本地服务器（127.0.0.1）的8000端口发起一个POST请求，请求的路径是"/infer/"后拼接上output的值
    data = response.json()    # 将响应的JSON内容解析为Python字典
    return (data["sample_rate"], np.array(data["audio"]))   # 从字典中取出"sample_rate"的值，返回采样率，同时将"audio"的值（一个列表）转换为numpy数组，并返回该数组

