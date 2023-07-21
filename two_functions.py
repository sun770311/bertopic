import csv
import pandas as pd
import os
import jieba
import re

from bertopic import BERTopic
from umap import UMAP
from sentence_transformers import SentenceTransformer

from nltk import flatten
from html2image import Html2Image


# REQUIRES: 
# 1) Original input csv file, 
# 2) Stopwords in txt file, 
# 3) Output csv file to store split sentences for bert() function
def split_input(filename, stopwords, splitfile):
    # Read the csv file
    my_var = open(filename).read().split()

    # Display first 10 rows 
    print(my_var[:10])

    # List of stopwords to exclude
    stopwords = [i.strip() for i in open(stopwords, "r", encoding = "utf-8").readlines()]

    # Define function to remove stopwords
    def pretty_cut(sentence):
        cut_list = jieba.lcut(''.join(re.findall('[\u4e00-\u9fa5]', sentence)), cut_all=True)
        for i in range(len(cut_list) - 1, -1, -1):
            if cut_list[i] in stopwords:
                del cut_list[i]
        return cut_list
    
    # Write processed text to output csv
    listoflists = []
    sublist = []
    with open(os.path.join(splitfile), "w", encoding = "utf-8", newline = '') as g:
        writer = csv.writer(g)
        for line in my_var:
            content = line.strip(" ")
            cut_content = " ".join(pretty_cut(content))
            writer.writerow([cut_content])
            sublist = []
            sublist.append(cut_content)
            listoflists.append(sublist)
    
    final_list = []
    for list_line in listoflists:
        for word in list_line:
            x = word.split(" ")
            final_list.append(x)
    
    # display first two split sentences in output csv 
    print(final_list[:2])


# REQUIRES: 
# 1) File with split sentences, 
# 2) Name of path to output visualization files
def bert(input_file, path_name):
    # used to embed sentences into a vector space
    model = SentenceTransformer('DMetaSoul/sbert-chinese-general-v2-distill')

    # Flatten csv file into one list
    input_list = pd.read_csv(input_file)
    input_list = input_list.fillna('')
    input_list = input_list.values.tolist()
    list_flat = flatten(input_list)

    # converts a string value into a collection of bytes
    embeddings = model.encode(list_flat)

    # Ensure that each run produces the same results 
    umap_model = UMAP(random_state=99)

    # Initializes topic model
    topic_model = BERTopic(language="chinese (simplified)",
                        umap_model= umap_model, # ensure same outcome each time
                        nr_topics = "auto", # automatically group together similar topics
                        top_n_words = 5, # display only the top 5 most frequent terms from each topic
                        calculate_probabilities = True, # calculates the probabilities of all topics across all documents instead of only the assigned topic
                        embedding_model = model, 
                        verbose = True)

    # fit the model on this list of documents and generate topics
    topics, probs = topic_model.fit_transform(list_flat, embeddings = embeddings)

    # Set path to input path when saving jpg
    hti = Html2Image(output_path = path_name)

    # 1. Topics
    topic_model.visualize_topics().write_html("topics.html")

    hti.screenshot(
        html_file='topics.html', 
        save_as='topics.jpg',
        size=(650, 650)
    )

    # 2. Distribution of topics
    topic_model.visualize_distribution(probs[100], min_probability=0.002).write_html('distribution.html')

    hti.screenshot(
        html_file='distribution.html', 
        save_as='distribution.jpg',
        size=(800, 600)
    )

    # 3. Topic hierarchy
    topic_model.visualize_hierarchy(top_n_topics=40).write_html('hierarchy.html')

    hti.screenshot(
        html_file='hierarchy.html', 
        save_as='hierarchy.jpg',
        size=(1000, 800)
    )

    # 4. Topic word scores
    topic_model.visualize_barchart(top_n_topics=8).write_html('barchart.html')

    hti.screenshot(
        html_file='barchart.html', 
        save_as='barchart.jpg',
        size=(1000, 500)
    )

    # 5. Topic similarity heatmap
    topic_model.visualize_heatmap(n_clusters=20, width=750, height=750).write_html('heatmap.html')

    hti.screenshot(
        html_file='heatmap.html', 
        save_as='heatmap.jpg',
        size=(750, 750)
    )


# Run functions
split_input('haodfcomments.csv', 'fixes-zh.txt', 'split_comments.csv')
bert('split_comments.csv', '/Users/hannahsun/Desktop/haodf')
