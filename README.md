# 📘 Nursery Admission Classification

Klasifikasi Kelayakan Penerimaan Anak di Nursery menggunakan **Machine Learning** dan **Deep Learning**.

---

## 👤 Informasi Proyek

- **Nama**: Haris Cahyana
- **NIM**: 233307050
- **Program Studi**: Teknologi Informasi
- **Repository**: [\[URL GitHub\]](https://github.com/HrsCode11/UAS-Data-Science.git)
- **Video Presentasi**: [\[URL Video\]](https://youtu.be/K8jEg3w__GI)

---

## 1. 🎯 Ringkasan Proyek

Proyek ini bertujuan untuk membangun sistem klasifikasi kelayakan penerimaan anak di nursery berdasarkan berbagai aspek sosial, ekonomi, dan kesehatan keluarga. Tahapan utama proyek meliputi:

- Exploratory Data Analysis (EDA)
- Data preparation dan preprocessing
- Pengembangan **3 model**:
  - **Model 1 (Baseline)**: Logistic Regression
  - **Model 2 (Advanced ML)**: Random Forest + Hyperparameter Tuning
  - **Model 3 (Deep Learning)**: Multilayer Perceptron (MLP)
- Evaluasi dan perbandingan performa model
- Penyimpanan model untuk reproducibility

---

## 2. 📄 Problem & Goals

### Problem Statements

- Bagaimana membangun model klasifikasi untuk menentukan kelayakan penerimaan anak di nursery?
- Model apa yang memberikan performa terbaik pada dataset Nursery?
- Apakah pendekatan deep learning mampu meningkatkan performa dibandingkan model klasik?

### Goals

- Membangun model klasifikasi dengan performa tinggi
- Membandingkan performa baseline, advanced ML, dan deep learning
- Menentukan model terbaik berdasarkan metrik evaluasi

---

## 3. 📁 Struktur Folder

```
project/
│
├── data/                   # Dataset Nursery (download manual)
│   └── nursery.data
│
├── notebooks/              # Notebook eksperimen
│   └── UAS.ipynb
│
├── models/                 # Model yang sudah dilatih
│   ├── logistic_regression_baseline.pkl
│   ├── random_forest_tuned.pkl
│   ├── mlp_nursery_model.keras
│
│
├── images/                 # Visualisasi (EDA & Evaluation)
│   ├── cm_logistic_regression.png
│   ├── cm_random_forest.png
│   ├── cm_mlp.png
│   ├── mlp_training_history.png
│   └── model_comparison.png
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```

---

## 4. 📊 Dataset

- **Sumber**: UCI Machine Learning Repository – Nursery Dataset
- **Jumlah Data**: 12.960 baris
- **Jumlah Fitur**: 8 fitur kategorikal
- **Target**: Kelas kelayakan nursery (multikelas)

### Fitur Utama

| Fitur    | Deskripsi         |
| -------- | ----------------- |
| parents  | Status orang tua  |
| has_nurs | Kondisi pengasuh  |
| form     | Struktur keluarga |
| children | Jumlah anak       |
| housing  | Kondisi rumah     |
| finance  | Kondisi keuangan  |
| social   | Kondisi sosial    |
| health   | Kondisi kesehatan |

---

## 5. 🔧 Data Preparation

- Tidak terdapat missing value atau duplikasi
- One-hot encoding untuk seluruh fitur kategorikal
- Label encoding untuk target
- StandardScaler digunakan untuk model MLP
- Data dibagi menggunakan **stratified split (80:20)**

---

## 6. 🤖 Modeling

- **Model 1 – Baseline**: Logistic Regression
- **Model 2 – Advanced ML**: Random Forest + GridSearchCV
- **Model 3 – Deep Learning**: Multilayer Perceptron (MLP)

---

## 7. 🧪 Evaluation

### Metrik Evaluasi

- Accuracy
- Precision (Macro Avg)
- Recall (Macro Avg)
- F1-Score (Macro Avg)

### Ringkasan Hasil

| Model               | Accuracy | F1-Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 0.93     | 0.89     |
| Random Forest       | 0.99     | 0.97     |
| MLP                 | 1.00     | 1.00     |

Model **MLP** memberikan performa terbaik pada dataset Nursery.

---

## 8. ▶️ Cara Menjalankan

### 8.1 Menjalankan di Google Colab (Direkomendasikan)

1. Upload file `UAS.ipynb` ke Google Colab
2. Upload dataset `nursery.data` ke folder `/data`
3. Jalankan seluruh cell dari atas ke bawah

### 8.2 Menjalankan Secara Lokal

1. Clone repository:

```bash
git clone <repo-url>
cd project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Jalankan notebook:

```bash
jupyter notebook notebooks/UAS.ipynb
```
