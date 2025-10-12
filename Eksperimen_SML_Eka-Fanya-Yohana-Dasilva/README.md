# Eksperimen Sistem Machine Learning (MSML) – Dicoding

## 🧠 Deskripsi
Repositori ini berisi eksperimen terhadap dataset pelatihan sebagai bagian dari submission akhir kelas **Membangun Sistem Machine Learning (MSML)** di Dicoding Indonesia.

Eksperimen dilakukan oleh:
**Eka Fanya Yohana Dasilva**  
NIM: 2218068  
Kelas: Membangun Sistem Machine Learning (MSML)

## 📊 Dataset
Dataset yang digunakan berasal dari **Kaggle**:
[Dataset for Indonesian Sentiment Analysis – Alvin Hanafie](https://www.kaggle.com/datasets/alvinhanafie/dataset-for-indonesian-sentiment-analysis)

Jumlah data awal: 11.000  
Jumlah data setelah pembersihan: 10.933  
Kolom: `text`, `sentiment`, `clean_text`

## 🧩 File dan Folder
- **namadataset_raw/** → berisi dataset mentah (`train_preprocess_ori.tsv`)
- **preprocessing/**  
  - `automate_EkaFanya.py` → script otomatisasi preprocessing  
  - `namadataset_preprocessing.csv` → hasil preprocessing  
- **Eksperimen_EkaFanya.ipynb** → notebook eksperimen manual sesuai template Dicoding

## ⚙️ Environment
- Python 3.12.7  
- pandas, nltk, re  
- mlflow==2.19.0 *(digunakan di tahap selanjutnya)*

## 📈 Status
✅ Tahap **Skilled (3 pts)** telah terpenuhi.  
Eksperimen, EDA, dan preprocessing telah dijalankan dengan sukses tanpa error.  
Script otomatisasi `automate_EkaFanya.py` menghasilkan output identik dengan eksperimen manual.

## 🏷️ Lisensi
© 2025 Eka Fanya Yohana Dasilva – Dicoding Indonesia
