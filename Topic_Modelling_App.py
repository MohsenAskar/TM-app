# Imports
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
from plotly.colors import qualitative
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.data import find
from gensim.models import LdaModel, HdpModel, CoherenceModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import spacy
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
import itertools
from nltk import ngrams
from docx import Document
from concurrent.futures import ThreadPoolExecutor
import time
import base64
import os

# Download necessary NLTK resources if not downloaded
def download_nltk_resources():
    try:
        find(r'tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        _ = stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
download_nltk_resources()

# Load SpaCy models with lazy loading
@st.cache_resource
def load_spacy_model(language):
    if language == 'english':
        return spacy.load('en_core_web_sm')
    elif language == 'norwegian':
        return spacy.load('nb_core_news_sm')

# Read .docx files
def read_docx(file):
    document = Document(file)
    return '\n'.join(paragraph.text for paragraph in document.paragraphs)

# Text preprocessing
@st.cache_data
def preprocess_text(text, language='english'):
    text = re.sub(r'\S*@\S*\s?', '', text)  
    nlp = load_spacy_model(language) 
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return tokens

# Clean text
@st.cache_data
def clean_text(text, language='english'):
    # Select stopwords and stemmer based on language
    stopwords_lang = set(stopwords.words(language))
    stemmer_lang = SnowballStemmer(language)

    if isinstance(text, list):
        return [clean_text(doc, language) for doc in text]

    # Lowercasing, removing special characters, numbers, and tokenizing
    text = re.sub(r'\W+', ' ', text.lower())
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)

    # Remove stopwords, perform stemming, and lemmatizing
    tokens = [stemmer_lang.stem(word) for word in tokens if word not in stopwords_lang]
    nlp = load_spacy_model(language)
    tokens = [token.lemma_ for token in nlp(' '.join(tokens))]

    return ' '.join(tokens)

# Calculate text statistics
@st.cache_data
def calculate_text_statistics(text, language):
    nlp = load_spacy_model(language)
    doc = nlp(text)
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    tokens = [token.text for token in doc if not token.is_punct]
    num_tokens = len(tokens)
    num_unique_words = len(set(tokens))
    words = [token.text for token in doc if not token.is_punct and not token.is_stop]
    num_words = len(words)
    stop_words = [token.text for token in doc if token.is_stop]
    num_stop_words = len(stop_words)
    avg_sentence_length = num_tokens / num_sentences if num_sentences else 0
    freq_dist = Counter(tokens)
    most_common_words = freq_dist.most_common(10)
    num_chars = len(text)
    avg_word_length = sum(len(word) for word in tokens) / len(tokens) if tokens else 0

    stats = [
        ('**Number of sentences**', num_sentences),
        ('**Number of tokens**', num_tokens),
        ('**Number of unique words**', num_unique_words),
        ('**Number of words**', num_words),
        ('**Number of stop words**', num_stop_words),
        ('**Average sentence length**', avg_sentence_length),
        ('**Most common words**', most_common_words),
        ('**Number of characters**', num_chars),
        ('**Average word length**', avg_word_length),
    ]
    
    return stats

# Perform topic modeling
@st.cache_data
def perform_topic_modeling(docs, num_topics=5, language='english'):
    tokens = [preprocess_text(doc, language) for doc in docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
    html_string = pyLDAvis.prepared_data_to_html(vis)
    return lda_model, html_string

# Calculate coherence score
@st.cache_data
def calculate_coherence_score(_lda_model, docs, language):
    language_code = "en" if language.lower() == "english" else "norwegian"
    tokens = [preprocess_text(doc, language) for doc in docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    coherence_model_lda = CoherenceModel(model=_lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
    return coherence_model_lda.get_coherence()

# t-SNE Visualization
def tsne_visualization(docs, language):
    n_samples = len(docs)
    if n_samples < 2:
        st.warning("Not enough data for t-SNE visualization.")
        return

    vectorizer = TfidfVectorizer(tokenizer=lambda x: preprocess_text(x, language), lowercase=False)
    X = vectorizer.fit_transform(docs)
    tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, n_samples - 1))
    X_2d = tsne.fit_transform(X.toarray())

    fig = px.scatter(
        x=X_2d[:, 0], 
        y=X_2d[:, 1],
        color=np.random.randint(0, 10, size=n_samples),
        color_continuous_scale='Plasma',
        labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
        title='t-SNE Visualization',
        hover_data={'Document': docs}
    )
    fig.update_traces(marker=dict(size=8, opacity=0.6))
    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig, use_container_width=True)

# Agglomerative Clustering
def agglomerative_clustering(docs, language):
    tokens = [preprocess_text(doc, language) for doc in docs]
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, preprocessor=lambda x: x)
    X = vectorizer.fit_transform(tokens)
    X_transposed = X.T.toarray()
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X_transposed)
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            current_count += 1 if child_idx < n_samples else counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    feature_names = vectorizer.get_feature_names_out()
    
    fig = plt.figure(figsize=(10, 10))
    dendrogram(linkage_matrix, labels=feature_names, leaf_rotation=90)
    st.pyplot(fig)

# HDP Topic Modeling
def hdp_topic_modeling(docs, language):
    tokens = [preprocess_text(doc, language) for doc in docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    hdp_model = HdpModel(corpus=corpus, id2word=dictionary)
    hdp_topics = hdp_model.show_topics(formatted=False)
    return hdp_model, hdp_topics

# Plot HDP Topics
def plot_topics(hdp_topics, title="Topic probabilities"):
    nrows, ncols = 5, 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(30, 20), sharey=False)
    axs = axs.flatten()
    cmap = plt.colormaps['tab20']
    colors = [cmap(i) for i in range(len(hdp_topics))]

    for ax, (topic_no, topic), color in zip(axs, hdp_topics, colors):
        words, probabilities = zip(*topic)
        x_pos = np.arange(len(words))
        ax.barh(x_pos, probabilities, color=color, edgecolor='black', linewidth=0.8)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(words, size=14)
        ax.invert_yaxis()
        ax.set_xlabel('Probability', size=12)
        ax.set_title(f'Topic #{topic_no}', size=14)

    for i in range(len(hdp_topics), nrows*ncols):
        fig.delaxes(axs[i])
    plt.suptitle(title, size=20)
    plt.tight_layout()
    return fig

# Perform DBSCAN Clustering
def perform_dbscan_clustering(docs, language):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: preprocess_text(x, language), lowercase=False)
    X = vectorizer.fit_transform(docs)
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    clusters = dbscan.fit_predict(X.toarray())
    pca = PCA(n_components=2)
    result = pca.fit_transform(X.toarray())
    unique_clusters = np.unique(clusters)

    # Choose a Plotly qualitative color scale
    color_scale = px.colors.qualitative.Vivid
    color_list = {str(cluster): color_scale[i % len(color_scale)] for i, cluster in enumerate(unique_clusters)}

    fig = px.scatter(
        x=result[:, 0],
        y=result[:, 1],
        color=clusters.astype(str),
        color_discrete_map=color_list,
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'color': 'Cluster'},
        title='DBSCAN Clustering',
        hover_data={'Document': docs}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.6), selector=dict(mode='markers'))
    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig)

# Show WordCloud
def show_wordcloud(docs, language):
    tokens = [token for doc in docs for token in preprocess_text(doc, language)]
    text = ' '.join(tokens)
    wordcloud = WordCloud(width=800, height=400).generate(text)
    fig = plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig)

# Show Word Frequency
def show_word_freq(docs, language):
    tokens = [token for doc in docs for token in preprocess_text(doc, language)]
    freq_dict = Counter(tokens)
    df = pd.DataFrame(freq_dict.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

    fig = px.bar(df.head(30), x='Frequency', y='Word', orientation='h', title="Word Frequency")
    st.plotly_chart(fig)

# Count and Visualize N-grams
def count_and_visualize_ngrams(docs, n, language):
    tokens = [preprocess_text(doc, language) for doc in docs]
    ngram_counts = [Counter(list(ngrams(doc_tokens, n))) for doc_tokens in tokens]
    combined_counts = sum(ngram_counts, Counter())
    ngrams_list = [' '.join(ngram) for ngram in combined_counts.keys()]
    frequencies = list(combined_counts.values())

    df = pd.DataFrame({'N-gram': ngrams_list, 'Frequency': frequencies}).sort_values('Frequency', ascending=False)
    df_top50 = df.head(50)
    
    fig, ax = plt.subplots()
    ax.bar(df_top50['N-gram'], df_top50['Frequency'])
    ax.set_xticklabels(df_top50['N-gram'], rotation=90, fontsize=6)
    ax.set_xlabel('N-gram')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Top 50 Most Frequent N-grams (n={n})')
    plt.tight_layout()
    st.pyplot(fig)

    return combined_counts

# Extract Keywords
def extract_keywords(docs, language):
    sentences = [sent_tokenize(doc) for doc in docs]
    sentences = list(itertools.chain(*sentences))

    vectorizer = TfidfVectorizer(tokenizer=lambda text: preprocess_text(text, language), lowercase=False)
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = X * X.T
    graph = from_scipy_sparse_matrix(similarity_matrix)
    scores = nx.pagerank(graph)
    ranked_keywords = sorted(scores, key=scores.get, reverse=True)

    feature_names = vectorizer.get_feature_names_out()
    return [feature_names[keyword] for keyword in ranked_keywords][:10]

# Create Graph from Sparse Matrix
def from_scipy_sparse_matrix(A, create_using=None):
    G = nx.Graph(create_using)
    coo = A.tocoo()
    G.add_nodes_from(range(A.shape[0]))
    G.add_weighted_edges_from(zip(coo.row, coo.col, coo.data))
    return G

# Convert the image to a base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Main Function
def main():
    image_path = (r"cartoon.JPG")
    image_base64 = image_to_base64(image_path)
    st.markdown(
        f"""
        <style>
        .header {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            padding: 10px;
            flex-direction: column; /* Stack items vertically */
        }}
        .header img {{
            border-radius: 50%;
            width: 50px;
            height: 50px;
            margin-bottom: 5px; /* Space between image and text */
        }}
        .header-text {{
            font-size: 16px;
            font-weight: normal; /* Regular weight for text */
            text-align: center;
        }}
        </style>
        <div class="header">
            <img src="data:image/jpeg;base64,{image_base64}" alt="Mohsen Askar">
            <div class="header-text">Developed by: Mohsen Askar</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Method Explanations")

    st.sidebar.markdown("### HDP (Hierarchical Dirichlet Process)")
    st.sidebar.markdown("""
    HDP is a non-parametric Bayesian approach to topic modeling that allows the number of topics to be inferred from the data. 
    It's useful for exploratory analysis when the number of topics isn't known beforehand.
    - [Read more about HDP](https://en.wikipedia.org/wiki/Hierarchical_Dirichlet_process)
    """)

    st.sidebar.markdown("### LDA (Latent Dirichlet Allocation)")
    st.sidebar.markdown("""
    LDA is a generative statistical model that discovers abstract topics within a collection of documents. 
    It requires specifying the number of topics and is widely used for understanding document themes.
    - [Read more about LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
    """)
    
    st.sidebar.markdown("### LDA Topic Coherence")
    st.sidebar.markdown("""
    Topic Coherence measures the semantic similarity between high scoring words in a topic. 
    Higher coherence scores typically indicate better topic quality, as they reflect the meaningfulness of the topics.
    - [Read more about LDA Topic Coherence](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)
    """)

    st.sidebar.markdown("### t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    st.sidebar.markdown("""
    t-SNE is a machine learning algorithm for dimensionality reduction, often used to visualize high-dimensional data.
    It helps in finding clusters and patterns by projecting data into a lower-dimensional space.
    - [Read more about t-SNE](https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a)
    """)

    st.sidebar.markdown("### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)")
    st.sidebar.markdown("""
    DBSCAN is a clustering algorithm that identifies dense regions of data points, separating them from sparser regions. 
    It's effective for finding clusters of varying shapes and handling noise.
    - [Read more about DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)
    """)

    st.sidebar.markdown("### N-grams")
    st.sidebar.markdown("""
    N-grams are contiguous sequences of n items from a given sample of text. 
    They are useful for capturing frequent phrases and patterns in text analysis.
    - [Read more about N-grams](https://en.wikipedia.org/wiki/N-gram)
    """)

    st.sidebar.markdown("### Keyword Extraction")
    st.sidebar.markdown("""
    Keyword extraction involves identifying the most relevant words or phrases in a text. 
    It's used for summarizing and understanding key topics and themes.
    - [Read more about Keyword Extraction](https://en.wikipedia.org/wiki/Keyword_extraction)
    """)

    st.title("Topic Modeling App (Experimental)")
    st.write("Upload text files or paste the text to perform text analysis and topic modeling.")

    # Language selection dropdown
    language = st.selectbox("Select Language", ["norwegian", "english"], help="Choose the language of your text")

    # File uploader
    uploaded_files = st.file_uploader("Upload text files", type=["txt", "docx"], accept_multiple_files=True, help= "You can upload one or many files at once")

    docs = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.txt'):
                text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith('.docx'):
                text = read_docx(uploaded_file)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue
            docs.extend(text.split('\n'))
            docs = [doc for doc in docs if doc.strip() != '']

            if docs:
                message_placeholder = st.empty()
                message_placeholder.info(f"Cleaning text from file {uploaded_file.name}, please wait...")
                cleaned_docs = [clean_text(doc, language) for doc in docs]
                message_placeholder.success(f"Cleaning text from file {uploaded_file.name} completed!")
                time.sleep(1)
                message_placeholder.empty()

    else:
        text = st.text_area("Or paste the text here")
        docs = text.split('\n')
        docs = [doc for doc in docs if doc.strip() != '']

        if docs:  
            message_placeholder = st.empty()
            message_placeholder.info("Cleaning text, please wait...")
            cleaned_docs = [clean_text(doc, language) for doc in docs]
            message_placeholder.success("Cleaning text is completed!")
            time.sleep(1)
            message_placeholder.empty()
                       
    DEFAULT_NGRAM = 2

    if st.button('Calculate Text Statistics'):
        if docs:
            results = calculate_text_statistics(' '.join(cleaned_docs), language)
            st.write("### Summary Statistics")
            for title, result in results:
                st.markdown(f'{title}: {result}')
        else:
            st.write("Please enter some text to calculate statistics.")
    
    if st.button("Analyze with HDP"):
        if cleaned_docs:
            with st.spinner('Starting HDP analysis...'):
                try:
                    hdp_model, hdp_topics = hdp_topic_modeling(cleaned_docs, language)
                    st.write('Total number of topics: ', len(hdp_model.get_topics()))
                    st.write('Topics found by HDP:')
                    for topic_no, topic in hdp_topics:
                        formatted_topic = ", ".join([f"{word} ({prob:.2f})" for word, prob in topic])
                        st.write(f"Topic #{topic_no}: {formatted_topic}")
                    st.session_state.hdp_topics = hdp_topics
                    st.success('HDP analysis completed successfully.')
                except Exception as e:
                    st.error(f'Error occurred during HDP analysis: {e}')
        else:
            st.warning("Please provide some text.")
            
    if st.button("Plot Topics of HDP"):
        if "hdp_topics" in st.session_state:
            try:
                st.write('Plotting topics...')
                fig = plot_topics(st.session_state.hdp_topics, title="Topic probabilities")
                st.pyplot(fig)
                st.write('Topics plotted successfully.')
            except Exception as e:
                st.error(f'Error occurred during plotting: {e}')
        else:
            st.warning("Please analyze the text first.")

    if st.button("Analyze with Agglomerative Clustering"):
        if cleaned_docs:
            with st.spinner('Performing Agglomerative Clustering...'):
                agglomerative_clustering(cleaned_docs, language)
        else:
            st.warning("Please provide some text.")

    num_topics = st.slider("Specify the number of Topics", min_value=1, max_value=50, value=2, help="Select the number of topics for LDA")

    if st.button("Analyze with LDA"):
        if cleaned_docs:
            with st.spinner('Performing LDA...'):
                lda_model, html_string = perform_topic_modeling(cleaned_docs, num_topics, language)
                st.components.v1.html(html_string, width=1200, height=1000, scrolling=True)
        else:
            st.warning("Please provide some text.")

    if st.button("Calculate Topic Coherence"):
        if cleaned_docs:
            with st.spinner('Calculating Coherence Score...'):
                lda_model, html_string = perform_topic_modeling(cleaned_docs, num_topics, language)
                coherence_score = calculate_coherence_score(lda_model, cleaned_docs, language.lower())
                st.write(f'Topic coherence: {coherence_score}')
        else:
            st.warning("Please provide some text.")

    if st.button("Visualize with t-SNE"):
        if cleaned_docs:
            with st.spinner('Performing t-SNE Visualization...'):
                tsne_visualization(cleaned_docs, language)
        else:
            st.warning("Please provide some text.")

    if st.button("Visualize with DBSCAN Clustering"):
        if cleaned_docs:
            with st.spinner('Performing DBSCAN Clustering...'):
                perform_dbscan_clustering(cleaned_docs, language)
        else:
            st.warning("Please provide some text.")

    if st.button("Generate WordCloud"):
        if cleaned_docs:
            with st.spinner('Generating WordCloud...'):
                show_wordcloud(cleaned_docs, language)
        else:
            st.warning("Please provide some text.")

    if st.button("Show Word Frequency"):
        if cleaned_docs:
            with st.spinner('Generating Word Frequency...'):
                show_word_freq(cleaned_docs, language)
        else:
            st.warning("Please provide some text.")

    n = st.number_input("Enter the value of n for n-grams", value=2, min_value=1, max_value=5, step=1, help="Select n for n-grams. N-grams are sequences of n words or tokens, often used to identify phrases or word patterns.")

    if st.button("Count and Visualize N-grams"):
        if cleaned_docs:
            with st.spinner('Counting N-grams...'):
                count_and_visualize_ngrams(cleaned_docs, n, language)
        else:
            st.warning("Please provide some text.")

    if st.button("Extract Keywords"):
        if cleaned_docs:
            with st.spinner('Extracting Keywords...'):
                keywords = extract_keywords(cleaned_docs, language)
                st.write("### Top Keywords")
                for keyword in keywords:
                    st.write(keyword)
        else:
            st.warning("Please provide some text.")

    if st.button("Compare LDA and HDP Models"):
        if cleaned_docs:
            lda_model, _ = perform_topic_modeling(cleaned_docs, num_topics, language)
            hdp_model, hdp_topics = hdp_topic_modeling(cleaned_docs, language)
            st.write("### LDA Topics")
            lda_topics = lda_model.show_topics(formatted=False)
            for topic_no, topic in lda_topics:
                formatted_topic = ", ".join([f"{word} ({prob:.2f})" for word, prob in topic])
                st.write(f"LDA Topic #{topic_no}: {formatted_topic}")

            st.write("### HDP Topics")
            for topic_no, topic in hdp_topics:
                formatted_topic = ", ".join([f"{word} ({prob:.2f})" for word, prob in topic])
                st.write(f"HDP Topic #{topic_no}: {formatted_topic}")
        else:
            st.warning("Please provide some text.")

if __name__ == "__main__":
    main()
