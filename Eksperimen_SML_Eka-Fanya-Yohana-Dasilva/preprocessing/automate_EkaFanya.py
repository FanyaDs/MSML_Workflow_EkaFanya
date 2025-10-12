import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import kagglehub

nltk.download("stopwords")

# =========================================================
# 1️⃣ Download dataset otomatis dari Kaggle
# =========================================================
print("📦 Mengunduh dataset dari Kaggle...")
dataset_path = kagglehub.dataset_download("alvinhanafie/dataset-for-indonesian-sentiment-analysis")
input_path = os.path.join(dataset_path, "train_preprocess_ori.tsv")

print("✅ Dataset ditemukan di:", input_path)

# =========================================================
# 2️⃣ Fungsi preprocessing
# =========================================================
def preprocess_dataset(input_path, output_path):
    df = pd.read_csv(input_path, sep="\t")
    print("📥 Data awal:", len(df), "baris")

    df = df.drop_duplicates(subset=["text"])
    print("🧹 Setelah hapus duplikat:", len(df), "baris")

    stop_words = set(stopwords.words("indonesian"))
    def clean_text(text):
        text = re.sub(r"[^a-zA-Z\s]", "", str(text))
        text = text.lower()
        tokens = [w for w in text.split() if w not in stop_words]
        return " ".join(tokens)

    df["clean_text"] = df["text"].apply(clean_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"💾 Dataset hasil preprocessing disimpan di: {output_path}")
    print("✅ Preprocessing selesai!")
    print(df.head())

# =========================================================
# 3️⃣ Jalankan preprocessing otomatis
# =========================================================
output_path = "preprocessing/namadataset_preprocessing/namadataset_preprocessing.csv"
preprocess_dataset(input_path, output_path)
