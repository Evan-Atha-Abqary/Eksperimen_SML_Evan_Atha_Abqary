import time
import psutil
import random
from prometheus_client import start_http_server, Gauge, Counter

# ==========================================
# DEFINISI METRIK (Total 11 Metrik Berbeda)
# ==========================================
# Metrik Infrastruktur Server
CPU_USAGE = Gauge('infra_cpu_usage_percent', 'Persentase penggunaan CPU')
MEM_USAGE_MB = Gauge('infra_memory_used_megabytes', 'Memori RAM yang digunakan dalam MB')
MEM_USAGE_PCT = Gauge('infra_memory_usage_percent', 'Persentase penggunaan RAM')

# Metrik Trafik Aplikasi & Model
TOTAL_REQUESTS = Counter('model_requests_total', 'Total request prediksi yang masuk')
SUCCESS_REQUESTS = Counter('model_requests_success_total', 'Total prediksi yang sukses')
FAILED_REQUESTS = Counter('model_requests_failed_total', 'Total prediksi yang gagal')
MODEL_LATENCY = Gauge('model_prediction_latency_seconds', 'Waktu respons prediksi model')

# Metrik Hasil Prediksi (Khas Tetouan City Power Consumption)
PREDICTED_POWER_ZONE1 = Gauge('power_predicted_zone1_kwh', 'Rata-rata hasil prediksi konsumsi listrik Zone 1')
PREDICTED_POWER_ZONE2 = Gauge('power_predicted_zone2_kwh', 'Rata-rata hasil prediksi konsumsi listrik Zone 2')
MODEL_ERROR_RATE = Gauge('model_error_rate_percent', 'Tingkat error simulasi pada model')
ACTIVE_USERS = Gauge('app_active_users_count', 'Jumlah pengguna aktif aplikasi saat ini')

def update_metrics():
    # Mengambil data asli dari laptop
    CPU_USAGE.set(psutil.cpu_percent())
    MEM_USAGE_MB.set(psutil.virtual_memory().used / (1024 * 1024))
    MEM_USAGE_PCT.set(psutil.virtual_memory().percent)
    
    # Simulasi trafik acak
    simulated_req = random.randint(1, 5)
    TOTAL_REQUESTS.inc(simulated_req)
    
    for _ in range(simulated_req):
        if random.random() > 0.05:
            SUCCESS_REQUESTS.inc()
        else:
            FAILED_REQUESTS.inc()
            
    MODEL_LATENCY.set(random.uniform(0.02, 0.45))
    PREDICTED_POWER_ZONE1.set(random.uniform(20000.0, 45000.0))
    PREDICTED_POWER_ZONE2.set(random.uniform(15000.0, 35000.0))
    MODEL_ERROR_RATE.set(random.uniform(1.2, 8.5))
    ACTIVE_USERS.set(random.randint(80, 150))

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus Exporter Berhasil Dijalankan di http://localhost:8000")
    while True:
        update_metrics()
        time.sleep(2)