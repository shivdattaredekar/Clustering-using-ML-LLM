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

![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/bcd7b1a2-62e3-47d8-bb70-8370c341c2a9)

## Text Embedding

Three methods are used for text embedding:
1. **TF-IDF (Term Frequency-Inverse Document Frequency)**: This method converts the text into numerical features based on the frequency of terms and their inverse document frequency.
![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/cf09d6ca-12a6-4c1a-8318-f62e363c3131)
   
2. **BERT (Bidirectional Encoder Representations from Transformers)**: A pre-trained language model that generates contextualized word embeddings.
![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/31395f94-717f-4bf3-8120-fd3a4dcdcd99)

3. **Sentence Transformers**: This model generates sentence-level embeddings using a pre-trained Sentence Transformer model.
![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/de682165-128a-4cdb-b015-fc9e3aa9abf7)

## Clustering

The optimal number of clusters is determined using the Elbow Method. The K-Means algorithm is then used to cluster the data into 20 clusters. The performance of the clustering is evaluated using the silhouette score.

![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/bd24f477-12ca-4520-bff6-e2838693d7da)

![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/4e88ce2d-49e2-4cbb-a027-2daa4accd81e)

![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/af2777e4-6b51-4163-91d9-b5e77d3ff3c5)

## Visualization

The clusters are visualized using PCA (Principal Component Analysis) to reduce the dimensionality of the embeddings. Scatter plots are generated to show the distribution of clusters. Additionally, bar charts are created to display the top 10 topics in each cluster.

## Naming Clusters with Google Gemini

The Google Gemini model is used to generate meaningful names for the clusters based on the email subjects.

## Results

- The TF-IDF model provided the highest silhouette score, indicating it was the most effective method for this dataset.
- Visualizations and cluster analysis were performed to gain insights into the email subjects.
Some examples of clusters :
![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/3fedfa2f-2fa5-4164-a87f-beb020e1b042)
![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/eab74d07-9c29-4908-8358-551706f1bb1d)
![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/0e9e3ad0-d0cd-4442-9e0a-bbb76a223757)
![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/988b8893-511e-4977-9830-cfbdcb42adb0)
![image](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/36d5b887-4855-4bb1-b536-850aa427b4e2)

- The Google Gemini model is utilized to name the clusters based on their subjects.
![Gemini Output](https://github.com/shivdattaredekar/Clustering-using-ML-LLM/assets/46707992/2e88434b-464b-41f6-bf15-bfc79b48ad98)


## Conclusion

The project successfully preprocesses and clusters email subjects using various text embedding techniques. The resulting clusters were visualized, and top topics within each cluster were identified.
