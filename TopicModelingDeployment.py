import streamlit as st
import pandas as pd
import pickle
import spacy
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt

# Set page config

st.set_page_config(
    page_title="News Topic Explorer",
    page_icon="📰",
    layout="wide"
)
nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat"])


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))

    res = [word
           for word in text
           if word not in stop_words
           and word[0].isalpha()]
    return ' '.join(res)


def replace_entities(text: str) -> str:
    """Replace entities in a single document"""
    ENTITY_LABELS = {"GPE", "ORG", "PERSON", "EVENT", "PRODUCT", "NORP"}
    doc = nlp(text)
    entities = sorted(
        [(ent.start_char, ent.end_char, ent.text)
         for ent in doc.ents
         if ent.label_ in ENTITY_LABELS and len(ent) > 1],
        key=lambda x: x[0],
        reverse=True
    )

    chars = list(text)
    for start, end, ent_text in entities:
        replacement = ent_text.replace(" ", "_")
        chars[start:end] = list(replacement)

    return "".join(chars)


def spacy_lemmatize(text: str) -> str:
    """Efficient lemmatization with spaCy"""
    doc = nlp(text)
    print(doc)
    return " ".join([token.lemma_ for token in doc if not token.is_space])


# Main app
def main():
    st.title("News Topic Modeling Explorer")

    # Sidebar controls
    st.sidebar.header("Controls")
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        ["NMF", "LDA", "LSA", "KMeans"]
    )

    tfidf = pickle.load(open('./NewsTopicModeling/tfidf_model.pkl', 'rb'))
    models =\
        {
            "NMF": pickle.load(open("./NewsTopicModeling/NMF_model (1).pkl", 'rb')),
            "LDA": pickle.load(open("./NewsTopicModeling/LDA_model.pkl", 'rb')),
            "LSA": pickle.load(open("./NewsTopicModeling/LSA_model.pkl", 'rb')),
            "KMeans": pickle.load(open("./NewsTopicModeling/KMeans_model.pkl", 'rb'))
        }

    # Main content area
    st.header(f"Model: {selected_model}")
    x = st.text_input("Input")
    if st.button("predict"):
        x = replace_entities(x)
        x = contractions.fix(x)
        x = word_tokenize(str.lower(x))
        x = remove_stop_words(x)
        x = spacy_lemmatize(x)
        x = tfidf.transform([x])

        output = models[selected_model].transform(x)
        if selected_model in ["NMF", "LDA", "LSA"]:
            # Probability-based models
            st.subheader("Topic Probabilities")

            components = models[selected_model].components_

            # Create topic names (you can customize these)
            topic_names = list()
            feature_names = tfidf.get_feature_names_out()
            num_top_words = 20
            for idx, comp in enumerate(components):
                top_indices = np.argsort(comp)[-num_top_words:][::-1]
                top_words = [feature_names[i]
                             for i in top_indices]
                entry = "_".join(top_words[:3])
                topic_names.append(entry)

            # Create and display probability table
            prob_df = pd.DataFrame({
                "Topic": topic_names,
                "Probability": output[0]  # First (and only) document's probabilities
            }).sort_values("Probability", ascending=False)

            st.table(prob_df.set_index(prob_df.columns[0]))

            # Visualize as bar chart
            st.bar_chart(prob_df.set_index("Topic"))

        elif selected_model == "KMeans":
            # Clustering model
            st.subheader("Cluster Assignment")
            components = models[selected_model].cluster_centers_

            # Create topic names (you can customize these)
            topic_names = list()
            topic_top_words = list()
            feature_names = tfidf.get_feature_names_out()
            num_top_words = 20
            for idx, comp in enumerate(components):
                top_indices = np.argsort(comp)[-num_top_words:][::-1]
                top_words = [feature_names[i]
                             for i in top_indices]
                entry = "_".join(top_words[:3])
                topic_names.append(entry)

            cluster_id = output[0].argmin()  # Get the cluster with highest score
            st.write(f"Assigned to Cluster: {topic_names[cluster_id]}")

            # Display cluster keywords (you'll need to define these)
            cluster_keywords = {
                0: ["north_korea", "missile", "nuclear", ...],
                1: ["say", "year", "one", ...],
                # ... define keywords for all clusters
            }

            # st.write("Cluster Keywords:")
            # st.write(", ".join(cluster_keywords.get(cluster_id, [])))

            # Show distance to all clusters
            st.subheader("Distance to All Clusters")
            dist_df = pd.DataFrame({
                "Cluster": topic_names,
                "Distance": output[0]
            }).sort_values("Distance")

            st.table(dist_df)
            st.bar_chart(dist_df.set_index("Cluster"))
    #     if selected_model == "NMF":
    #         topics = nmf_results
    #     elif selected_model == "LDA":
    #         topics = lda_results
    # data = {"hello": 0.7}
    #
    # # Convert to DataFrame and rename the column
    # df = pd.DataFrame({"Topics": ["syria", "terror"], "Probablility": [0.7, 0.4]})
    #
    # # Hide the index and display the table
    # st.dataframe(df.set_index(df.columns[0]))

    # # Show metrics
    # st.subheader("Model Performance")
    # model_metrics = metrics[metrics['model'] == selected_model.lower()]
    # st.dataframe(model_metrics.style.highlight_max(axis=0))
    #
    # # Topic visualization
    # st.subheader("Topic Visualization")
    #
    # if selected_model == "NMF":
    #     topics = nmf_results
    # elif selected_model == "LDA":
    #     topics = lda_results
    # else:
    #     topics = pd.DataFrame()  # Add other models
    #
    # if not topics.empty:
    #     col1, col2 = st.columns([1, 3])
    #
    #     with col1:
    #         selected_topic = st.selectbox(
    #             "Select Topic",
    #             topics['Topic_ID'].unique()
    #         )
    #
    #     with col2:
    #         topic_words = topics[topics['Topic_ID'] == selected_topic]['Top_Words'].values[0]
    #
    #         # Create word cloud
    #         wordcloud = WordCloud(width=800, height=400).generate(topic_words)
    #         plt.figure(figsize=(10, 5))
    #         plt.imshow(wordcloud)
    #         plt.axis("off")
    #         st.pyplot(plt)
    #
    # # Document samples
    # st.subheader("Example Documents")
    # if selected_model == "BERTopic":
    #     doc_info = bert_model.get_document_info(df['processed_content'])
    #     st.dataframe(doc_info.head(10))
    # else:
    #     st.dataframe(df[['title', 'publication', 'processed_content']].head(10))
    #
    # # Model comparison section
    # st.subheader("Model Comparison")
    # st.bar_chart(metrics.set_index('model')['coherence'])


# Run the app
if __name__ == "__main__":
    main()