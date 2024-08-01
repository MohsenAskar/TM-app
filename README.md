# TM-app
Topic Modeling and Text Analysis App
## Overview:
This app performs some Natural Language Processing (NLP) techniques including topic modeling and text analysis on English and Norwegian texts. It's designed for users who want to gain insights into textual data by exploring underlying themes, identifying key terms, and visualizing word distributions. 
## Features:
- ### Text Preprocessing:
  Automatically cleans and processes text by removing special characters, stopwords, and performing lemmatization and stemming.
- ### Topic Modeling:
#### HDP (Hierarchical Dirichlet Process): 
  Infers the number of topics from the data, suitable for exploratory analysis.
#### LDA (Latent Dirichlet Allocation): 
  Discovers a specified number of abstract topics from a collection of documents.
- ### Visualization:
#### t-SNE: 
  Visualizes high-dimensional data to uncover patterns and clusters.
#### DBSCAN Clustering: 
  Identifies dense regions and separates them from noise, effectively finding clusters of varying shapes.
#### WordCloud: 
  Generates a visual representation of the most frequent words in the text.
- ### Text Statistics:
  Provides summary statistics including the number of sentences, words, tokens, and common words.
- ### N-grams Analysis:
  Extracts and visualizes frequent phrases or word patterns in the text.
- ### Keyword Extraction:
  Identifies the most relevant keywords and key phrases using the TextRank algorithm.

## How It Works
1. Input Text: Upload a text file or paste the text into the provided text area.
2. Select Language: Choose between English or Norwegian to tailor the analysis.
3. Analyze: Use buttons to trigger different analyses, such as topic modeling with HDP or LDA, visualizations with t-SNE or DBSCAN, and more.
4. Interactive Results: View detailed results and interactive visualizations that provide insights into the structure and key themes of the text.
  
