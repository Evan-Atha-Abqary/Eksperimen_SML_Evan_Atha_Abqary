import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

# Load data preprocessed (Alamat sudah disesuaikan karena CSV sejajar dengan script)
df = pd.read_csv("dataset_preprocessed.csv")
X = df.drop("zone_1_power_consumption", axis=1)
y = df["zone_1_power_consumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Autolog dimatikan untuk mencegah error saat berjalan di server GitHub
mlflow.sklearn.autolog(disable=True)

if __name__ == "__main__":
    with mlflow.start_run(run_name="CI_Automated_Training") as run:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan model secara lokal agar bisa diambil oleh GitHub Actions
        mlflow.sklearn.log_model(model, "model")
        
        # Menuliskan Run ID ke file agar mudah dibaca oleh Docker build step
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print("Model berhasil dilatih via MLflow Project di GitHub Actions!")