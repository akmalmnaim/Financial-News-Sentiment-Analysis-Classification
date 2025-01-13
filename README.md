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
- Menerapkan teknik augmentasi data untuk meningkatkan jumlah data pada kelas yang kurang terwakili.

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari file CSV yang berisi informasi terkait analisis sentimen pada berita keuangan yang tersedia di situs web [Kaggle](https://www.kaggle.com). Dataset tersebut adalah dataset kualitatif yang berisi judul berita keuangan yang memiliki label sentimen neutrl,positif, dan negatif memiliki total 4846 baris dan 2 kolom data.

- **Jumlah Baris Data**: 4846 baris
- **Kondisi Data**: Data terdiri dari dua kolom utama, yaitu sentimen dan dokumen yang berupa judul berita finansial. Setiap baris mewakili satu berita dengan label sentimen yang terkait.

Dataset ini cocok digunakan untuk membangun model supervised learning dalam NLP (Natural Language Processing) multi class classification. Dalam kasus ini adalah untuk mengklasifikasikan berita finansial dengan sentimen neutral, positif, negatif

[Kaggle - Financial News Sentiment Analysis](https://www.kaggle.com/code/khotijahs1/nlp-financial-news-sentiment-analysis/input).

### Kolom/Fitur Data:

1. **Sentimen**:

   - Kolom ini berisi label sentimen dari berita keuangan.
   - Terdapat 3 kategori sentimen:
     - **Positif**: Berita keuangan yang memberikan sentimen optimis atau menggembirakan.
     - **Negatif**: Berita keuangan yang mengandung sentimen pesimis atau buruk.
     - **Netral**: Berita keuangan yang tidak cenderung memiliki dampak positif maupun negatif secara jelas.

2. **Judul Berita Finansial**:
   - Kolom ini berisi teks atau judul dari berita keuangan yang dianalisis sentimennya. Judul-judul ini merupakan bahan analisis untuk mengklasifikasikan sentimen terkait.

### Variabel-variabel pada dataset:

- **Sentiment**: Merupakan label kelas yang menunjukkan sentimen berita (positif, negatif, netral).
- **Document**: Merupakan isi berita keuangan yang akan dianalisis.

**_Exploratory Data Analysis_**

Exploratory Data Analysis (EDA) adalah pendekatan analisis data yang bertujuan untuk memahami karakteristik utama dari kumpulan data. EDA melibatkan penggunaan teknik statistik dan visualisasi grafis untuk menemukan pola, hubungan, atau anomali untuk membentuk hipotesis. Proses ini sering kali tidak terstruktur dan dianggap sebagai langkah awal penting dalam analisis data yang membantu menentukan arah analisis lebih lanjut.

Berikut ini adalah EDA yang dilakukan :

- ```python
  print(df['Sentiment'].value_counts())
  ```

  Kode diatas memiliki output:

  ```python
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 4846 entries, 0 to 4845
  Data columns (total 2 columns):
  #   Column     Non-Null Count  Dtype
  ---  ------     --------------  -----
   0   Sentiment  4846 non-null   object
   1   Document   4846 non-null   object
  dtypes: object(2)
  memory usage: 75.8+ KB
  None
  ```

  Kolom Sentiment:

  Non-Null Count: 4846 entri non-null, artinya semua baris di kolom ini memiliki data, tidak ada nilai yang hilang atau NaN.
  Dtype: object, menunjukkan bahwa tipe data di kolom ini adalah string (teks). Kolom ini kemungkinan berisi label sentimen seperti "positive", "neutral", atau "negative".

  Kolom Document:

  Non-Null Count: 4846 entri non-null, yang berarti setiap baris memiliki teks dokumen tanpa nilai kosong.
  Dtype: object, menunjukkan bahwa tipe data di kolom ini adalah string. Kolom ini berisi teks atau kalimat yang merupakan isi dari dokumen yang akan dianalisis.

- ```python
    print(df['Sentiment'].value_counts())
  ```
  Kode diatas memiliki output:
  ```python
    Sentiment
   neutral     2879
   positive    1363
   negative     604
   Name: count, dtype: int64
  ```
  Terdapat 3 kelas sentimen yaitu
  -Terdapat 2879 dokumen dalam dataset yang memiliki sentimen netral.

-Terdapat 1363 dokumen yang memiliki sentimen positif.

-Terdapat 604 dokumen dengan sentimen negatif.

**Visualisasi Data**

- _Univariate Analysis_

  _Univariate Analysis_ adalah jenis analisis data yang memeriksa satu variabel (atau bidang data) pada satu waktu. Tujuannya adalah untuk menggambarkan data dan menemukan pola yang ada dalam distribusi variabel tersebut. Ini termasuk penggunaan statistik deskriptif, histogram, dan box plots untuk menganalisis distribusi dan memahami sifat dari variabel tersebut.

  gambar 1

Elemen-Elemen dalam Grafik:

Sumbu X: Menampilkan kategori sentimen, yaitu:
Netral: Sentimen yang tidak menunjukkan emosi positif atau negatif yang kuat.
Negatif: Sentimen yang menunjukkan emosi negatif, seperti ketidakpuasan, kemarahan, atau kekecewaan.
Positif: Sentimen yang menunjukkan emosi positif, seperti kesenangan, kepuasan, atau harapan.
Sumbu Y: Menampilkan jumlah atau frekuensi dari setiap kategori sentimen. Semakin tinggi batang, semakin banyak data yang termasuk dalam kategori tersebut.

## Data Preparation

Dalam tahap ini, beberapa teknik persiapan data diterapkan, termasuk penghapusan data yang tidak relevan, preprocessing teks, dan augmentasi data.

### Proses Data Preparation

1. Menghapus dokumen kosong dan melakukan preprocessing untuk membersihkan teks.

- ```python
  print(df.isnull().sum())
  ```
  Kode diatas memiliki output:
  ```python
    Sentiment    0
    Document     0
    dtype: int64
  ```

didapatkan data tidak memiliki missing values

2. melakukan preprocessing untuk membersihkan teks
```python

      # Preprocessing function
      
      def preprocess_text(text): # Mengubah teks menjadi huruf kecil
      text = text.lower()
      
                  # Menghapus tanda baca, angka, dan karakter spesial
                  text = re.sub(r'[^a-z\s]', '', text)
      
                  # Tokenisasi
                  tokens = word_tokenize(text)
      
                  # Menghapus stopwords
                  stop_words = set(stopwords.words('english'))
                  tokens = [word for word in tokens if word not in stop_words]
      
                  # Lemmatization (atau bisa gunakan stemming)
                  lemmatizer = WordNetLemmatizer()
                  tokens = [lemmatizer.lemmatize(word) for word in tokens]
      
                  # Menggabungkan kembali token menjadi kalimat
                  return ' '.join(tokens)
      
              # Terapkan preprocessing ke kolom Document
              df['cleaned_Document'] = df['Document'].apply(preprocess_text)
      
              # Menampilkan beberapa data setelah preprocessing
              print(df[['Document', 'cleaned_Document']].head())
```
```python
               Document  \
        0  According to Gran , the company has no plans t...
        1  Technopolis plans to develop in stages an area...
        2  The international electronic industry company ...
        3  With the new production plant the company woul...
        4  According to the company 's updated strategy f...
      
                                            cleaned_Document
        0  according gran company plan move production ru...
        1  technopolis plan develop stage area le square ...
        2  international electronic industry company elco...
        3  new production plant company would increase ca...
        4  according company updated strategy year baswar...
```

penjelasan
a. Pengubahan Huruf Kecil:
Teks diubah menjadi huruf kecil untuk menghindari perbedaan antara huruf besar dan kecil, sehingga konsistensi dalam analisis dapat terjaga.
b. Penghapusan Tanda Baca, Angka, dan Karakter Spesial:
Menggunakan regular expressions (regex), semua karakter yang bukan huruf dan spasi dihapus. Ini membantu membersihkan teks dari elemen yang tidak relevan.
c. Tokenisasi:
Teks dibagi menjadi kata-kata (token) menggunakan fungsi word_tokenize. Ini adalah langkah penting sebelum melakukan analisis lebih lanjut.
d. Penghapusan Stopwords:
Stopwords, yaitu kata-kata umum seperti "the", "is", "and", yang tidak membawa makna signifikan, dihapus dari daftar token. Ini membantu fokus pada kata-kata yang lebih berarti dalam konteks analisis.
e. Lematisasi:
Setiap token diproses melalui lemmatizer untuk mengubah kata-kata ke bentuk dasarnya. Lematisasi berbeda dari stemming karena menghasilkan kata yang lebih tepat dan dapat dikenali dalam konteks bahasa.
f. Penggabungan Kembali Token:
Setelah pemrosesan, token yang sudah dibersihkan digabungkan kembali menjadi kalimat untuk disimpan dalam kolom baru cleaned_Document.

3. Fungsi Augmentasi Data Menggunakan Sinonim

```python
  def synonym_augment(text):
    words = text.split()
    augmented_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            augmented_words.append(synonym.replace('_', ' '))
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)
```

Kode diatas memiliki output:

```python
 def augment_data(df, target_class_size):
  class_counts = df['Sentiment'].value_counts()
  new_data = []

  for sentiment in class_counts.index:
      while class_counts[sentiment] < target_class_size:
          example = df[df['Sentiment'] == sentiment].sample().iloc[0]
          augmented_example = synonym_augment(example['cleaned_Document'])
          new_data.append({'cleaned_Document': augmented_example, 'Sentiment': sentiment})
          class_counts[sentiment] += 1

  augmented_df = pd.DataFrame(new_data)
  return pd.concat([df, augmented_df], ignore_index=True)
```

a. Fungsi `synonym_augment`
Fungsi ini bertujuan untuk meningkatkan variasi teks dengan mengganti kata-kata dalam kalimat dengan sinonimnya. Berikut adalah langkah-langkah yang dilakukan:

- **Input**: Menerima parameter `text`, yang merupakan kalimat yang ingin diaugmentasi.
- **Tokenisasi**: Memecah kalimat menjadi kata-kata menggunakan `split()`.
- **Pencarian Sinonim**:
  - Untuk setiap kata, fungsi mencari sinonim menggunakan `wordnet.synsets(word)`.
  - Jika sinonim ditemukan, salah satu sinonim dipilih secara acak.
  - Sinonim yang dipilih ditambahkan ke dalam daftar `augmented_words` setelah mengganti underscore (`_`) dengan spasi.
- **Pengembalian**: Menggabungkan kembali kata-kata yang telah diaugmentasi menjadi string dan mengembalikannya.

b. Fungsi `augment_data`
Fungsi ini digunakan untuk menambah jumlah contoh dalam dataset sehingga setiap kelas sentimen memiliki ukuran yang seimbang. Langkah-langkahnya adalah sebagai berikut:

- **Input**: Menerima `data` (DataFrame dengan kolom 'cleaned_Document' dan 'Sentiment') dan `target_class_size`, ukuran target untuk setiap kelas sentimen.
- **Hitung Kelas**: Menghitung jumlah contoh untuk setiap kelas sentimen menggunakan `value_counts()`.
- **Augmentasi Data**:
  - Untuk setiap kelas sentimen, selama jumlah contoh kelas tersebut kurang dari `target_class_size`, ambil satu contoh acak dari kelas tersebut.
  - Terapkan `synonym_augment` pada contoh yang diambil dan tambahkan hasilnya ke dalam daftar `new_data`.
  - Perbarui jumlah kelas yang dihitung.
- **Pengembalian**: Mengembalikan DataFrame baru yang berisi data augmentasi.

Fungsi-fungsi ini sangat berguna dalam meningkatkan ukuran dataset dan mendukung keseimbangan antara kelas-kelas yang berbeda dalam model pembelajaran mesin.

## Model Development

Model Machine Learning yang digunakan adalah Support Vector Classification (SVC). Setiap model dilatih dengan menggunakan data yang telah diproses dan dievaluasi dengan metrik yang sesuai. Menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) yang merupakan metode untuk menilai seberapa penting sebuah kata dalam dokumen relatif terhadap seluruh koleksi dokumen.

### Kelebihan dan Kekurangan:

- **Support Vector Classification (SVC)**:

  - **Kelebihan**: SVC efektif untuk data berdimensi tinggi dan tetap bekerja baik meski jumlah dimensi lebih besar dari jumlah sampel.
  - \*_Kekurangan_: SVC bisa lambat pada dataset besar

- **Keunggulan TF-IDF:**
  -Mengurangi pengaruh kata umum (stop words).
  -Menekankan kata yang relevan dan jarang.
  -Efisien untuk dataset besar.
  -Membantu mengurangi overfitting.
  -Cocok untuk analisis teks dan klasifikasi.

## Evaluation

# Evaluasi

Setelah model dibangun dan diuji dengan data uji, perlu dilakukan evaluasi untuk menilai kinerja model. Dalam evaluasi model klasifikasi multi-kelas (tiga label), metrik yang digunakan adalah `Accuracy`, `Precision`, `Recall`, dan `F1 Score` yang diperoleh dari _Confusion Matrix_.

_Confusion Matrix_ adalah tabel yang digunakan untuk mengevaluasi kinerja model klasifikasi. Dalam klasifikasi multi-kelas, matriks ini menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas, dengan membaginya ke dalam beberapa kategori:

- **True Positives (TP)**:
  Jumlah prediksi benar untuk suatu kelas tertentu. Contoh: Model memprediksi kelas "A" dan kelas sebenarnya juga "A".

- **True Negatives (TN)**:
  Jumlah prediksi benar yang bukan untuk kelas tertentu. Contoh: Model tidak memprediksi kelas "A" untuk kasus yang sebenarnya bukan "A".

- **False Positives (FP)** (Type I Error):
  Jumlah prediksi salah di mana model memprediksi suatu kelas tertentu, tetapi sebenarnya bukan kelas itu. Contoh: Model memprediksi kelas "A" tetapi kelas sebenarnya adalah "B" atau "C".

- **False Negatives (FN)** (Type II Error):
  Jumlah prediksi salah di mana model gagal memprediksi kelas tertentu yang sebenarnya benar. Contoh: Model memprediksi kelas "B" atau "C" sementara kelas sebenarnya adalah "A".

### Metrik Evaluasi

Berikut adalah penjelasan dari metrik yang digunakan dalam konteks multi-kelas:

- **`Accuracy`**:
  \[
  Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
  \]
  Mengukur seberapa sering model memprediksi dengan benar di seluruh kelas. Dalam multi-kelas, akurasi adalah jumlah prediksi benar (semua TP) dibagi dengan jumlah total kasus.

- **`Precision`** (untuk setiap kelas \(i\)):
  \[
  Precision_i = \frac{TP_i}{TP_i + FP_i}
  \]
  Mengukur proporsi prediksi untuk kelas \(i\) yang benar-benar kelas \(i\). Dalam multi-kelas, presisi dihitung untuk setiap kelas secara individu.

- **`Recall`** (untuk setiap kelas \(i\)):
  \[
  Recall*i = \frac{TP_i}{TP_i + FN_i}
  \]
  Mengukur proporsi kasus aktual dari kelas \(i\) yang teridentifikasi dengan benar. \_Recall* juga dihitung untuk setiap kelas.

- **`F1 Score`** (untuk setiap kelas \(i\)):
  \[
  F1 Score_i = \frac{2 \cdot Precision_i \cdot Recall_i}{Precision_i + Recall_i}
  \]
  F1 Score adalah rata-rata harmonik dari presisi dan recall untuk setiap kelas, memberikan keseimbangan antara keduanya.

Untuk multi-kelas, kita biasanya menghitung _macro average_ dan _weighted average_ dari metrik-metrik ini:

- **Macro Average**: Rata-rata sederhana dari metrik untuk semua kelas.
- **Weighted Average**: Rata-rata tertimbang berdasarkan jumlah sampel di setiap kelas.

Berikut adalah hasil evaluasi model menggunakan metrik `Accuracy`, `Precision`, `Recall`, dan `F1 Score` dari _Confusion Matrix_.

```python
 # Untuk setiap model
    for model_name, model in models.items():
        # Latih model
        model.fit(X_train_vec, y_train)

        # Prediksi dan evaluasi
        predictions = model.predict(X_test_vec)
        report = classification_report(y_test, predictions, output_dict=True)

        # Menampilkan classification report
        print(f"Classification Report for {model_name}:")
        print(f"Accuracy: {report['accuracy']}")
        print(f"Precision (Macro Average): {report['macro avg']['precision']}")
        print(f"Recall (Macro Average): {report['macro avg']['recall']}")
        print(f"F1 Score (Macro Average): {report['macro avg']['f1-score']}")



```

Berikut ini adalah hasilnya:

```python
Classification Report for SVC:
Accuracy: 0.8697916666666666
Precision (Macro Average): 0.870067375372337
Recall (Macro Average): 0.8693722510214398
F1 Score (Macro Average): 0.8690521254757266

Classification Report for Random Forest:
Accuracy: 0.8356481481481481
Precision (Macro Average): 0.8381387719648762
Recall (Macro Average): 0.8349906184921992
F1 Score (Macro Average): 0.8341249430503991

```

Berikut ini adalah Visualisasi dari `Confusion Matrix`:

![Confusion Matrix - Final](https://private-user-images.githubusercontent.com/50210408/402446244-4cc6525b-f888-473c-86f9-7e3d213b4a0c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzY3NDk5MjcsIm5iZiI6MTczNjc0OTYyNywicGF0aCI6Ii81MDIxMDQwOC80MDI0NDYyNDQtNGNjNjUyNWItZjg4OC00NzNjLTg2ZjktN2UzZDIxM2I0YTBjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAxMTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMTEzVDA2MjcwN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWFhMGRjNzc4Y2YwZWU4ZDdkOWIzYjk5NjU5ZDJhNWQ3ZDU3ZTliN2U5ZTNkNWM5NzA4NmQ2NmFmNWYxZjQ2OTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.8FtzWkNcAVNZK0jUl090pTFut54ZTDr1aZFDoedGLe4)

<div align="center">Gambar 7b - Confusion Matrix Results</div>

<br>

Berdasarkan visualisasi data diatas, hasilnya dapat dirincikan sebagai berikut:
-True Positive (TP):
-510 data yang sebenarnya netral berhasil diprediksi sebagai netral.
-454 data yang sebenarnya positif berhasil diprediksi sebagai positif.
-True Negative (TN): 539 data yang sebenarnya negatif berhasil diprediksi sebagai negatif.
-False Positive (FP):
-20 data yang sebenarnya negatif salah diprediksi sebagai netral.
-16 data yang sebenarnya negatif salah diprediksi sebagai positif.
-52 data yang sebenarnya netral salah diprediksi sebagai positif.
-83 data yang sebenarnya positif salah diprediksi sebagai netral.
-False Negative (FN):
-18 data yang sebenarnya netral salah diprediksi sebagai negatif.
-32 data yang sebenarnya positif salah diprediksi sebagai negatif.
-Interpretasi:

-Model cukup baik dalam mengklasifikasikan data negatif: Sebagian besar data negatif berhasil diklasifikasikan dengan benar.
-Model juga cukup baik dalam mengklasifikasikan data positif: Sebagian besar data positif berhasil diklasifikasikan dengan benar.
-Model masih kesulitan dalam mengklasifikasikan data netral: Terdapat cukup banyak kesalahan klasifikasi pada data netral, baik diklasifikasikan sebagai negatif maupun positif.

### Hasil Proyek Berdasarkan Metrik Evaluasi pada model Terbaik

- **Model**: SVC
- **Vektorisasi**: TF-IDF
- **Akurasi**: 0.8692129629629629
- **Presisi**: 0.8697920346464035
- **Recall**: 0.8687854353447709
- **F1 Score**: 0.8685468903767801

Dengan menggunakan 2 model, hasil yang diperoleh menunjukkan bahwa model SVC dengan augmentasi data secara signifikan meningkatkan performa model pada kelas yang kurang terwakili.

## Kesimpulan

Proyek ini menunjukkan pentingnya analisis sentimen dalam memahami dampak berita keuangan terhadap pasar. Dengan menerapkan teknik Machine Learning, kita dapat mengklasifikasikan sentimen berita dengan baik dan menyediakan alat bantu bagi para investor untuk membuat keputusan yang lebih baik.

---

Catatan:

- Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan.
- Jika terdapat penjelasan yang harus menyertakan kode, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode proyek, cukup bagian yang ingin dijelaskan saja.
