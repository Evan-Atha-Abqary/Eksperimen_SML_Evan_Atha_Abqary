import pandas as pd
import os

def load_data(file_path):
    print(f"Memuat data dari: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Melakukan preprocessing data...")
    df_processed = df.copy()
    
    # Transformasi DateTime dan ekstraksi fitur waktu
    df_processed['DateTime'] = pd.to_datetime(df_processed['DateTime'])
    df_processed['Month'] = df_processed['DateTime'].dt.month
    df_processed['Hour'] = df_processed['DateTime'].dt.hour
    df_processed['DayOfWeek'] = df_processed['DateTime'].dt.dayofweek
    
    # Menghapus kolom lama dan merapikan nama kolom
    df_processed = df_processed.drop('DateTime', axis=1)
    df_processed.columns = df_processed.columns.str.replace(' ', '_').str.lower()
    
    return df_processed

def save_data(df, output_path):
    # Membuat folder tujuan jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data preprocessed berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    # Path relatif ini menyesuaikan posisi saat script dijalankan dari luar folder
    RAW_DATA_PATH = "data_raw/Tetuan_City_power_consumption.csv"
    PROCESSED_DATA_PATH = "data_preprocessing/dataset_preprocessed.csv"
    
    try:
        # Menjalankan urutan fungsi
        df_raw = load_data(RAW_DATA_PATH)
        df_clean = preprocess_data(df_raw)
        save_data(df_clean, PROCESSED_DATA_PATH)
        print("Proses Automatisasi Selesai Tanpa Error!")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")