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
from sklearn.metrics import classification_report, confusion_matrix
import random

# A. Business Understanding
## 1. Memahami Struktur Dataset
df = pd.read_csv("all-data.csv", delimiter=',', encoding='latin-1', names=['Sentiment', 'Document'])
print(df.info())
print(df['Sentiment'].value_counts())

## 2. Visualisasi Distribusi Sentimen
sns.countplot(x="Sentiment", data=df)
plt.title('Distribusi Sentimen')
plt.show()



# B. Data Preparation
## Tahapan Preprocessing Teks
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


## 1. Fungsi Augmentasi Data Menggunakan Sinonim
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


## 2. Evaluasi Model untuk Analisis Sentimen
def evaluate_model(data, augment=False):
    # Split data menjadi training dan testing
    X = data['cleaned_Document']
    y = data['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Jika augmentasi diperlukan
    if augment:
        target_class_size = max(y.value_counts())
        data = augment_data(data, target_class_size)  # Melakukan augmentasi
        X_train, X_test, y_train, y_test = train_test_split(data['cleaned_Document'], data['Sentiment'], test_size=0.2, random_state=42)

    # Gunakan TF-IDF untuk vektorisasi teks
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Gunakan SVC sebagai model
    model = SVC()
    model.fit(X_train_vec, y_train)

    # Prediksi dan evaluasi
    predictions = model.predict(X_test_vec)
    report = classification_report(y_test, predictions, output_dict=True)

    
    
    # Menampilkan classification report
    print(f"Classification Report:")
    print(f"Accuracy: {report['accuracy']}")
    print(f"Precision (Macro Average): {report['macro avg']['precision']}")
    print(f"Recall (Macro Average): {report['macro avg']['recall']}")
    print(f"F1 Score (Macro Average): {report['macro avg']['f1-score']}")

    # Menampilkan confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Visualisasi confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Menyimpan model terbaik
    with open('best_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return report
