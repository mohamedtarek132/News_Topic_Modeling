# News Topic Modeling
Dataset Link: https://www.kaggle.com/datasets/mohamedtarek01234/articles 


This project explores various topic modeling techniques applied to a collection of news articles. The primary objective is to extract meaningful and coherent topics from unstructured text data using unsupervised machine learning methods. The project encompasses the entire pipeline: from data preprocessing and vectorization to model training, evaluation, and deployment.
## Project Overview
As part of my academic exploration into natural language processing and unsupervised learning, I implemented and compared several topic modeling algorithms:

**Latent Dirichlet Allocation (LDA)**

**Non-negative Matrix Factorization (NMF)**

**Latent Semantic Analysis (LSA)**

**KMeans Clustering**

The dataset underwent preprocessing steps including tokenization, lemmatization, and stop word removal. Subsequently, the text was transformed into numerical features using TF-IDF vectorization. Each model was trained and evaluated based on coherence scores and topic-word distributions.

## Repository Contents
news-topic-modeling (5).ipynb: Jupyter Notebook detailing preprocessing steps, model training, topic extraction, and visualizations.

TopicModelingDeployment.py: Script for deploying trained models on new text data.

Model Files:

- LDA_model.pkl

- NMF_model (1).pkl

- LSA_model.pkl

- KMeans_model.pkl

- tfidf_model.pkl

- News Topic Modeling Documentation.pdf: Comprehensive report detailing methodology, results, and key insights.

## Technologies Used
Programming Language: Python

Libraries:

- Natural Language Processing: spaCy, nltk

- Topic Modeling & Machine Learning: scikit-learn, gensim

- Data Manipulation: pandas, numpy

- Visualization: matplotlib, seaborn

## Results & Evaluation
The performance of each topic modeling technique was evaluated using coherence scores:

### Latent Dirichlet Allocation (LDA):

Coherence Score: 0.45

Produced coherent topics with overlapping themes.

### Non-negative Matrix Factorization (NMF):

Coherence Score: 0.50

Generated distinct and interpretable topics.

### Latent Semantic Analysis (LSA):

Coherence Score: 0.40

Captured broader themes but with less specificity.

### KMeans Clustering:

Coherence Score: 0.35

Identified clusters based on term frequency but lacked topic depth.

### Summary

Based on these evaluations, NMF demonstrated the highest coherence score, indicating its effectiveness in producing distinct and interpretable topics for this dataset. Detailed analyses and visualizations are provided in the News Topic Modeling Documentation.pdf.
