import numpy as np
import requests

def run(output):
    response = requests.post(f"http://127.0.0.1:8000/infer/{output}")
    data = response.json()
    return (data["sample_rate"], np.array(data["audio"]))

