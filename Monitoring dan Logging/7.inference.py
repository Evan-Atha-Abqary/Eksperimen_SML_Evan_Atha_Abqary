import requests
import time
import random

URL = "http://127.0.0.1:8000"

print("Memulai simulasi pengiriman trafik ke model...")
while True:
    try:
        response = requests.get(URL)
        if response.status_code == 200:
            print(f"Ping Model Berhasil! Status: {response.status_code}")
    except Exception as e:
        print("Menunggu Exporter Aktif...")
    time.sleep(random.uniform(0.5, 1.5))