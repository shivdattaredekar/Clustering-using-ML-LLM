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

## Overview

This project involves the following steps:
1. Data import and cleaning
2. Text preprocessing
3. Text embedding using TF-IDF, BERT, and Sentence Transformers
4. Clustering the embedded vectors
5. Visualizing the clusters
6. Naming the clusters using the Google Gemini model

## Data Import and Preprocessing

The data is imported from an Excel file and checked for missing and non-text values. Missing values are dropped, and non-text rows are filtered out. The text data is then preprocessed by removing special characters, converting to lowercase, removing stopwords, and lemmatizing the words.

## Text Embedding

Three methods are used for text embedding:
1. **TF-IDF (Term Frequency-Inverse Document Frequency)**: This method converts the text into numerical features based on the frequency of terms and their inverse document frequency.
2. **BERT (Bidirectional Encoder Representations from Transformers)**: A pre-trained language model that generates contextualized word embeddings.
3. **Sentence Transformers**: This model generates sentence-level embeddings using a pre-trained Sentence Transformer model.

## Clustering

The optimal number of clusters is determined using the Elbow Method. The K-Means algorithm is then used to cluster the data into 20 clusters. The performance of the clustering is evaluated using the silhouette score.

## Visualization

The clusters are visualized using PCA (Principal Component Analysis) to reduce the dimensionality of the embeddings. Scatter plots are generated to show the distribution of clusters. Additionally, bar charts are created to display the top 10 topics in each cluster.

## Naming Clusters with Google Gemini

The Google Gemini model is used to generate meaningful names for the clusters based on the email subjects.

## Results

- The TF-IDF model provided the highest silhouette score, indicating it was the most effective method for this dataset.
- Visualizations and cluster analysis were performed to gain insights into the email subjects.
- The Google Gemini model is utilized to name the clusters based on their subjects.

## Conclusion

The project successfully preprocesses and clusters email subjects using various text embedding techniques. The resulting clusters were visualized, and top topics within each cluster were identified.
