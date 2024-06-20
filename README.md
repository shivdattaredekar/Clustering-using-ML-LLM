# Email Subject Clustering and Embedding

This project aims to preprocess, clean, and cluster email subjects from a dataset, leveraging various text embedding techniques such as TF-IDF, BERT, and Sentence Transformers.

## Table of Contents
- [Overview](#overview)
- [Data Import and Preprocessing](#data-import-and-preprocessing)
- [Text Embedding](#text-embedding)
- [Clustering](#clustering)
- [Visualization](#visualization)
- [Naming Clusters with Google Gemini](#naming-clusters-with-google-gemini)
- [Results](#results)

# Overview

This project involves the following steps:
1. Data import and cleaning
2. Text preprocessing
3. Text embedding using TF-IDF, BERT, and Sentence Transformers
4. Clustering the embedded vectors
5. Visualizing the clusters
6. Naming the clusters using the Google Gemini model

## Data Import and Preprocessing

We start by importing necessary libraries and the dataset:

```python
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

# Importing the data
df = pd.read_excel('Email Dataset.xlsx')

# Checking the data
df.head(20)
df.shape

# Check for the missing values in the data
print(f'Total Missing values in the data : {df.isnull().sum()}')

# Drop the missing values
df = df.dropna().reset_index().drop(columns={'index'})
print(f'Size of data after dropping missing values : {df.shape}')

# Check for non-text values
non_string_rows = df[df['subject'].apply(lambda x: isinstance(x, int))]
print(f'Rows that only have numbers and non text values:{non_string_rows.shape}')
df = df[~ (~ df['subject'].apply(lambda x: isinstance(x, str)))].reset_index().drop(columns={'index'})
print(f'Size of data after dropping non-text rows : {df.shape}')

# Check for the missing values in the data
print(f'Total Missing values in the data : {df.isnull().sum()}')

# Drop the missing values
df = df.dropna().reset_index().drop(columns={'index'})
print(f'Size of data after dropping missing values : {df.shape}')

# Check for non-text values
non_string_rows = df[df['subject'].apply(lambda x: isinstance(x, int))]
print(f'Rows that only have numbers and non text values:{non_string_rows.shape}')
df = df[~ (~ df['subject'].apply(lambda x: isinstance(x, str)))].reset_index().drop(columns={'index'})
print(f'Size of data after dropping non-text rows : {df.shape}')

nltk.download('stopwords')
nltk.download('wordnet')
stopwords.words('english')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Data preprocessing
df1 = df['subject'].tolist()
corpus = []

for i in range(len(df1)):
    review = re.sub('[^a-zA-Z]', ' ', df1[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

data = pd.DataFrame({'subject': corpus})
data = data[data['subject'] != ''].reset_index().drop(columns={'index'})
print(data.shape)

# Text Embedding
We use three methods for text embedding: TF-IDF, BERT, and Sentence Transformers.

## TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
X = vectorizer.fit_transform(data['subject']).toarray()

# Print example
print("Document:", corpus[3])
print("TF-IDF representation:")
print(X[3])

## BERT
from transformers import BertTokenizer, BertModel
import torch
import time

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to process email subjects in batches
def process_batch(email_subjects):
    encoded_inputs = tokenizer(email_subjects, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings

# Process email subjects in batches
batch_size = 1000
start_time = time.time()
cls_embeddings_list = []

for i in range(0, len(data), batch_size):
    batch_subjects = data['subject'].iloc[i:i+batch_size].tolist()
    cls_embeddings = process_batch(batch_subjects)
    cls_embeddings_list.append(cls_embeddings)

cls_embeddings_all = torch.cat(cls_embeddings_list, dim=0).numpy()
end_time = time.time()
print("Elapsed time: {:.2f} seconds".format(end_time - start_time))

X_cls_bert = np.array(cls_embeddings_all)

## Sentence Transformers
import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
st = time.time()

data['sentence_transformers'] = data['subject'].apply(lambda text: model.encode(text, convert_to_numpy=True))

et = time.time()
print("Elapsed time: {:.2f} seconds".format(et - st))

X_transformers = np.vstack(data['sentence_transformers'].values)

#clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_optimal_clusters(data, max_k):
    iters = range(1, max_k + 1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('Elbow Method For Optimal k')
    plt.show()

find_optimal_clusters(X, 30)
find_optimal_clusters(X_transformers, 30)
find_optimal_clusters(X_cls_bert, 30)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def eval_cluster(embeddings, labels):
    score = silhouette_score(embeddings, labels)
    print(f'Silhouette Score: {score:.4f}')

def dimension_reduction(embedding, method):
    pca = PCA(n_components=2, random_state=42)
    pca_vecs = pca.fit_transform(embedding)
    if len(pca_vecs) == len(data):
        data[f'x0_{method}'] = pca_vecs[:, 0]
        data[f'x1_{method}'] = pca_vecs[:, 1]
    else:
        raise ValueError(f"Length of PCA vectors ({len(pca_vecs)}) does not match length of DataFrame ({len(data)})")

def plot_pca(x0, x1, cluster_name, method):
    plt.figure(figsize=(10, 7))
    plt.scatter(x=data[x0], y=data[x1], c=data[cluster_name], cmap='viridis', s=50, alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA Plot of Clusters - {method}')
    plt.colorbar()
    plt.show()

for embedding_and_method in [(X, 'tfidf'), (X_transformers, 'transformers'), (X_cls_bert, 'bert')]:
    embedding, method = embedding_and_method[0], embedding_and_method[1]
    kmeans = KMeans(n_clusters=20, random_state=42)
    kmeans.fit(embedding)
    clusters = kmeans.labels_
    clusters_result_name = f'cluster_{method}'
    data[clusters_result_name] = clusters
    eval_cluster(embedding, clusters)
    dimension_reduction(embedding, method)
    plot_pca(f'x0_{method}', f'x1_{method}', cluster_name=clusters_result_name, method=method)
    plt.savefig(f'{method}_clusters.png')

#Visualization
import matplotlib.pyplot as plt

cluster_groups = data.groupby('cluster_tfidf')

for cluster, group in cluster_groups:
    topic_counts = group['subject'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    topic_counts.head(10).plot(kind='bar', color='skyblue')
    plt.title(f'Top 10 Topics in Cluster {cluster}')
    plt.xlabel('Topics')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'cluster_{cluster}_top_10_topics.png')
    plt.show()

#Naming Clusters with Google Gemini
import google.generativeai as genai

GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def get_response(input_text, prompt):
    response = model.generate_content([input_text, prompt])
    return response.text

input_text = "\n".join(data['subject'].tolist())
prompt = "Please name the following clusters based on their email subjects."
response = get_response(input_text, prompt)
print(response)

#Results
  - The TF-IDF model provided the highest silhouette score, indicating it was the most effective method for this dataset.
  - Visualizations and cluster analysis were performed to gain insights into the email subjects.
  - Google Gemini Model will be used to name the clusters based on their subjects.
