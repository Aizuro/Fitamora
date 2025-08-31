import os
import pandas as pd

# Ganti path dengan lokasi dataset di komputer lokal
dataset_path = "posture_dataset"  # Contoh: "C:/Users/User/dataset_postur"

# Ambil semua nama folder (label)
labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# Simpan data
data = []

# Loop tiap folder
for label in labels:
    folder = os.path.join(dataset_path, label)
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(folder, file)
            # Buat row: filepath + one-hot encoding untuk label
            row = [filepath] + [1 if l == label else 0 for l in labels]
            data.append(row)

# Buat DataFrame
df = pd.DataFrame(data, columns=["filepath"] + labels)

# Simpan ke CSV
df.to_csv("dataset.csv", index=False)

print("CSV multi-label berhasil dibuat! Jumlah data:", len(df))
print("Labels:", labels)