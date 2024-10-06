import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import download

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import random
import pickle
import joblib

# A. Business Understanding
## 1. Memahami Struktur Dataset
df = pd.read_csv("all-data.csv", delimiter=',', encoding='latin-1', names=['Sentiment', 'Document'])
print(df.info())
print(df['Sentiment'].value_counts())

## 2. Visualisasi Distribusi Sentimen
sns.countplot(x="Sentiment", data=df)
plt.title('Distribusi Sentimen')
plt.show()

## 3. Analisis Panjang Dokumen
df['doc_length'] = df['Document'].apply(lambda x: len(x.split()))
print(df['doc_length'].describe())
sns.histplot(df['doc_length'], bins=20)
plt.title('Distribusi Panjang Dokumen')
plt.show()

# B. Data Preparation
## 1. Mengatasi Missing Values
print(df.isnull().sum())

## 2. Memeriksa Baris Kosong
print(df['Document'].isnull().sum())

## 3. Memeriksa Baris yang Hanya Berisi Spasi
print(df['Document'].apply(lambda x: len(str(x).strip()) == 0).sum())

## 4. Menghapus Baris Tidak Valid
df = df[df['Document'].apply(lambda x: len(str(x).strip()) > 0)]

## 5. Tahapan Preprocessing Teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned_Document'] = df['Document'].apply(preprocess_text)
print(df[['Document', 'cleaned_Document']].head())

# C. Data Modeling
## 1. Deteksi dan Penghapusan Outlier
Q1 = df['doc_length'].quantile(0.25)
Q3 = df['doc_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['doc_length'] < lower_bound) | (df['doc_length'] > upper_bound)]
print("Outlier berdasarkan panjang dokumen:")
print(outliers[['cleaned_Document', 'doc_length']])

def remove_outliers(data):
    return data[(data['doc_length'] >= lower_bound) & (data['doc_length'] <= upper_bound)]

df_no_outliers = remove_outliers(df)
print(f"Original DataFrame shape: {df.shape}")
print(f"New DataFrame shape after outlier removal: {df_no_outliers.shape}")

## 2. Fungsi Augmentasi Data Menggunakan Sinonim
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

## 3. Model Klasifikasi
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
 "Random Forest": RandomForestClassifier(),
    "SVC": SVC(),
    "Naive Bayes": MultinomialNB()
}

scenarios = [
    {"name": "With Outlier + No Data Augment", "data": df},
    {"name": "No Outlier + No Data Augment", "data": df_no_outliers},
    {"name": "With Outlier + Data Augment", "data": None},
    {"name": "No Outlier + Data Augment", "data": None}
]

## 4. Evaluasi Model untuk Analisis Sentimen
def evaluate_models(scenarios):
    results = []
    best_model = None
    best_f1_score = 0
    for scenario in scenarios:
        data = scenario["data"]
        if "Data Augment" in scenario["name"]:
            target_class_size = max(data['Sentiment'].value_counts())
            data = augment_data(data, target_class_size)
        X = data['cleaned_Document']
        y = data['Sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizers = {
            'TF-IDF': TfidfVectorizer(),
            'Count Vectorizer': CountVectorizer()
        }
        for vectorizer_name, vectorizer in vectorizers.items():
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            for model_name, model in models.items():
                model.fit(X_train_vec, y_train)
                predictions = model.predict(X_test_vec)
                report = classification_report(y_test, predictions, output_dict=True)
                results.append({
                    'Scenario': scenario['name'],
                    'Model': model_name,
                    'Accuracy': report['accuracy'],
                    'Precision': report['macro avg']['precision'],
                    'Recall': report['macro avg']['recall'],
                    'F1 Score': report['macro avg']['f1-score'],
                    'Vectorizer': vectorizer_name
                })
                if report['macro avg']['f1-score'] > best_f1_score:
                    best_f1_score = report['macro avg']['f1-score']
                    best_model = model
    return results, best_model

scenarios_no_augment = [
    {"name": "With Outlier + No Data Augment", "data": df},
    {"name": "No Outlier + No Data Augment", "data": df_no_outliers}
]

scenarios_with_augment = [
    {"name": "With Outlier + Data Augment", "data": df},
    {"name": "No Outlier + Data Augment", "data": df_no_outliers}
]

results_no_augment, best_model_no_augment = evaluate_models(scenarios_no_augment)
results_with_augment, best_model_with_augment = evaluate_models(scenarios_with_augment)

results = results_no_augment + results_with_augment

best_model = best_model_no_augment if best_model_no_augment else best_model_with_augment

if best_model is not None:
    with open('best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("Model terbaik disimpan dalam format pickle sebagai best_model.pkl")

## 5. Penyimpanan Model
with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

joblib.dump(best_model, 'best_model.joblib')
print("Model terbaik disimpan dalam format Joblib sebagai best_model.joblib")

## 6. Visualisasi Hasil Evaluasi Model
results_df = pd.DataFrame(results)
print(results_df)

scenario_vectorizer_df = results_df[['Scenario', 'Vectorizer']].drop_duplicates()
print(scenario_vectorizer_df)

results_df['Scenario_Vectorizer'] = results_df['Scenario'] + ' - ' + results_df['Vectorizer']

plt.figure(figsize=(20, 20))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i + 1)
    sns.barplot(data=results_df, x='Scenario_Vectorizer', y=metric, hue='Model')
    plt.title(f'Model {metric} Comparison')
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.xlabel('Scenario_Vectorizer')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

## 7. Pencarian Model Terbaik
best_model = results_df.loc[results_df['F1 Score'].idxmax()]
print("\nBest Performing Model:")
print(f"Model: {best_model['Model']}")
print(f"Scenario: {best_model['Scenario']}")
print(f"Vectorizer: {best_model['Vectorizer']}")
print(f"Accuracy: {best_model['Accuracy']:.4 f}")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall: {best_model['Recall']:.4f}")
print(f"F1 Score: {best_model['F1 Score']:.4f}")

if best_model['F1 Score'] >= 0.8:
    print("The model performed exceptionally well, indicating a strong predictive capability.")
elif best_model['F1 Score'] >= 0.6:
    print("The model performed moderately well, indicating potential for improvement.")
else:
    print("The model's performance is below expectations, suggesting that further optimization or data preprocessing may be required.")