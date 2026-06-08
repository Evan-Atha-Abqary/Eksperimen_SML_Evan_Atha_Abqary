import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

# Mengaktifkan Autolog untuk kriteria Basic
mlflow.sklearn.autolog()

# Load data preprocessed
df = pd.read_csv("dataset_preprocessed.csv")
X = df.drop("zone_1_power_consumption", axis=1)
y = df["zone_1_power_consumption"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    with mlflow.start_run(run_name="Basic_RandomForest_Autolog"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Kriteria Basic: model berhasil dilatih dengan autolog!")