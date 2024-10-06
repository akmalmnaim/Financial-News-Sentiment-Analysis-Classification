# Laporan Proyek Machine Learning - Akmal Muhammad Naim

## Domain Proyek

Proyek ini bertujuan untuk melakukan analisis sentimen terhadap berita keuangan dengan menggunakan metode Machine Learning. Sentimen dalam konteks ini merujuk pada pandangan positif, negatif, atau netral yang terkandung dalam teks berita. Mengingat perkembangan pesat dalam industri keuangan, analisis sentimen dapat memberikan wawasan yang berharga tentang bagaimana berita memengaruhi pasar dan keputusan investasi.

### Mengapa masalah ini harus diselesaikan:

- Sentimen pasar yang terdistorsi dapat memengaruhi keputusan investasi dan berpotensi menyebabkan kerugian finansial.
- Dengan menggunakan analisis sentimen, investor dapat mengambil keputusan yang lebih tepat berdasarkan data yang tersedia.

**Referensi:**

- Sentiment Analysis in Financial Markets: A Survey

## Business Understanding

### Problem Statements

1. Bagaimana cara mengklasifikasikan berita keuangan ke dalam kategori sentimen positif, negatif, dan netral?
2. Apakah ada pola dalam panjang dokumen berita yang dapat mempengaruhi akurasi analisis sentimen?
3. Bagaimana cara mengatasi ketidakseimbangan kelas dalam dataset sentimen berita keuangan?

### Goals

1. Mengembangkan model Machine Learning yang dapat mengklasifikasikan sentimen berita keuangan dengan akurasi tinggi.
2. Menganalisis distribusi panjang dokumen untuk memahami hubungannya dengan klasifikasi sentimen.
3. Mengimplementasikan teknik augmentasi data untuk meningkatkan representasi kelas yang kurang terwakili dalam dataset.

### Solution Statements

- Menggunakan beberapa algoritma, seperti Logistic Regression, Random Forest, Naive Bayes, dan Support Vector Classification (SVC) untuk mencapai solusi yang diinginkan.
- Menerapkan teknik augmentasi data untuk meningkatkan jumlah data pada kelas yang kurang terwakili dan melakukan hyperparameter tuning pada model untuk meningkatkan performa.

## Data Understanding

Data yang digunakan dalam proyek ini diperoleh dari file CSV yang berisi informasi tentang sentimen dan dokumen berita keuangan. Anda dapat mengunduh dataset dari Situs Kaggle.

Berikut Linknya :
https://www.kaggle.com/code/khotijahs1/nlp-financial-news-sentiment-analysis/input

### Variabel-variabel pada dataset:

- **Sentiment**: Merupakan label kelas yang menunjukkan sentimen berita (positif, negatif, netral).
- **Document**: Merupakan isi berita keuangan yang akan dianalisis.

## Data Preparation

Dalam tahap ini, beberapa teknik persiapan data diterapkan, termasuk penghapusan data yang tidak relevan, preprocessing teks, dan augmentasi data.

### Proses Data Preparation

1. Membaca dataset dan memeriksa informasi dasar.
2. Menghapus dokumen kosong dan melakukan preprocessing untuk membersihkan teks.
3. Menerapkan augmentasi data untuk meningkatkan jumlah data terutama data pada kelas yang kurang terwakili.

## Model Development

Model Machine Learning yang digunakan untuk menganalisis sentimen meliputi Logistic Regression, Random Forest, Naive Bayes, dan Support Vector Classification (SVC). Setiap model dilatih dengan menggunakan data yang telah diproses dan dievaluasi dengan metrik yang sesuai.

### Kelebihan dan Kekurangan:

- **Logistic Regression**:

  - **Kelebihan**: Sederhana dan cepat.
  - **Kekurangan**: Tidak bekerja baik dengan data yang tidak linear.

- **Random Forest**:

  - **Kelebihan**: Robust dan tidak mudah overfitting.
  - **Kekurangan**: Lebih lambat dalam prediksi dibandingkan model lainnya.

- **Naive Bayes**:

  - **Kelebihan**: Cepat dan efisien.
  - **Kekurangan**: Asumsi independensi antar fitur tidak selalu valid.

- **Support Vector Classification (SVC)**:
  - **Kelebihan**: SVC efektif untuk data berdimensi tinggi dan tetap bekerja baik meski jumlah dimensi lebih besar dari jumlah sampel.
  - \*_Kekurangan_: SVC bisa lambat pada dataset besar dan memerlukan tuning parameter yang tepat untuk performa optimal.

Proyek ini terdiri dari 32 skenario, yang mencakup variasi dari dua pendekatan utama (TFIDF dan Count Vectorizer) dan empat perlakuan data yang berbeda. Berikut adalah variasinya:

- **Scenario (TFIDF)**:

  - With outlier + no data augment
  - No outlier + no data augment
  - With outlier + data augment (train data)
  - No outlier + data augment (train data)

- **Scenario (Count Vectorizer)**:

  - With outlier + no data augment
  - No outlier + no data augment
  - With outlier + data augment (train data)
  - No outlier + data augment (train data)

- **Model**:
  - Logistic Regression
  - Random Forest
  - SVC
  - Naive Bayes

## Evaluation

Metrik evaluasi yang digunakan dalam proyek ini mencakup akurasi, precision, recall, dan F1 score. Hasil analisis menunjukkan bahwa model yang digunakan dapat memberikan akurasi yang memuaskan.

### Hasil Proyek Berdasarkan Metrik Evaluasi pada model terbaik

- **Model**: SVC
- **Skenario**: Dengan Outlier + Data Augment
- **Vektorisasi**: TF-IDF
- **Akurasi**: 0.8808
- **Presisi**: 0.8806
- **Recall**: 0.8804
- **F1 Score**: 0.8802

Dengan menggunakan berbagai skenario, hasil yang diperoleh menunjukkan bahwa augmentasi data secara signifikan meningkatkan performa model pada kelas yang kurang terwakili.

## Kesimpulan

Proyek ini menunjukkan pentingnya analisis sentimen dalam memahami dampak berita keuangan terhadap pasar. Dengan menerapkan teknik Machine Learning, kita dapat mengklasifikasikan sentimen berita dengan baik dan menyediakan alat bantu bagi para investor untuk membuat keputusan yang lebih baik.

---

