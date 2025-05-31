# News Topic Modeling

**Dataset Link**: [https://www.kaggle.com/datasets/mohamedtarek01234/articles](https://www.kaggle.com/datasets/mohamedtarek01234/articles)

This project explores various topic modeling techniques applied to a collection of news articles. The primary objective is to extract meaningful and coherent topics from unstructured text data using unsupervised machine learning methods. The project encompasses the entire pipeline: from data preprocessing and vectorization to model training, evaluation, and deployment.

## Project Overview

As part of my academic exploration into natural language processing and unsupervised learning, I implemented and compared several topic modeling algorithms:

- **Latent Dirichlet Allocation (LDA)**
- **Non-negative Matrix Factorization (NMF)**
- **Latent Semantic Analysis (LSA)**
- **KMeans Clustering**

The dataset underwent preprocessing steps including tokenization, lemmatization, and stop word removal. Subsequently, the text was transformed into numerical features using TF-IDF vectorization. Each model was trained and evaluated based on coherence scores and topic-word distributions.

## Preprocessing

The preprocessing pipeline included the following steps to ensure clean and standardized input:

* Named entities were detected using spaCy and preserved as single tokens by replacing spaces with underscores.
* Contractions were expanded using the `contractions` library to ensure consistency in tokenization.
* Text was converted to lowercase, tokenized with NLTK, and cleaned by removing stopwords and non-alphabetic tokens.
* Non-English articles were filtered out using `langdetect`.
* Words were lemmatized using spaCy to reduce them to their base forms.
* The `url` column was dropped, and articles with fewer than 50 words were excluded.

## Repository Contents

* `news-topic-modeling (5).ipynb`: Jupyter Notebook detailing preprocessing steps, model training, topic extraction, and visualizations.
* `TopicModelingDeployment.py`: Script for deploying trained models on new text data.

Model Files:

* `LDA_model.pkl`
* `NMF_model (1).pkl`
* `LSA_model.pkl`
* `KMeans_model.pkl`
* `tfidf_model.pkl`
* `News Topic Modeling Documentation.pdf`: Comprehensive report detailing methodology, results, and key insights.

## Technologies Used

**Programming Language**: Python

**Libraries**:

* Natural Language Processing: spaCy, nltk
* Topic Modeling & Machine Learning: scikit-learn, gensim
* Data Manipulation: pandas, numpy
* Visualization: matplotlib, seaborn

## Results & Evaluation

The performance of each topic modeling technique was evaluated using coherence scores and additional clustering metrics:

| **Model**  | **Coherence (c\_v)** | **Diversity** | **Exclusivity** | **Perplexity** | **Silhouette Score** |
| ---------- | -------------------- | ------------- | --------------- | -------------- | -------------------- |
| **NMF**    | 0.686                | 89.5%         | 0.952           | N/A            | N/A                  |
| **LDA**    | 0.590                | 94.0%         | 0.976           | 32,869.8       | N/A                  |
| **LSA**    | 0.384                | 34.5%         | 0.589           | N/A            | N/A                  |
| **KMeans** | 0.642                | 77.0%         | 0.932           | N/A            | 0.008                |

### Latent Dirichlet Allocation (LDA)
Achieved high diversity and exclusivity, but lower coherence and high perplexity indicate overlapping and less distinguishable topics.

### Non-negative Matrix Factorization (NMF)
Delivered the highest coherence score and strong exclusivity, making it the most effective in producing clear and interpretable topics.

### Latent Semantic Analysis (LSA)
Performed poorly across all metrics, indicating broad and less focused topics.

### KMeans Clustering
Showed decent coherence and exclusivity but suffered from low silhouette score, suggesting weak cluster separation.


### Summary

Based on these evaluations, NMF demonstrated the highest coherence score, indicating its effectiveness in producing distinct and interpretable topics for this dataset. Detailed analyses and visualizations are provided in the `News Topic Modeling Documentation.pdf`.
