import json
import csv
import pandas as pd
import numpy as np
import os
import jieba
import re

# 1. PRE-PROCESSING

# Read in input file 
my_var = open('haodfcomments.csv').read().split()

# Display first 10 entries of input
print(my_var[:10])

# Select stopwords (words lacking significance to strip from input)
stopwords = [i.strip() for i in open('./baidu1.txt', "r", encoding = 
                                     "utf-8").readlines()]

# Function to strip stopwords
def pretty_cut(sentence):
    cut_list = jieba.lcut(''.join(re.findall('[\u4e00-\u9fa5]', sentence)), cut_all=True)
    for i in range(len(cut_list) - 1, -1, -1):
        if cut_list[i] in stopwords:
            del cut_list[i]
    return cut_list

# Write stripped input to new csv file hello1
with open(os.path.join("./hello1.csv"), "w", encoding = "utf-8", newline = '') as g:
    writer = csv.writer(g)
    writer.writerow(["content"])
    for line in my_var:
        content = line.strip(" ")
        cut_content = " ".join(pretty_cut(content))
        writer.writerow([cut_content])

# 2. BERTopic TRANSFORMATIONS

from bertopic import BERTopic

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

input_file = pd.read_csv('hello1.csv', usecols = ['content'])
input_file = input_file.fillna('')
input_file = input_file.values.tolist()

from nltk import flatten
my_input = flatten(input_file)

embeddings = model.encode(my_input)

def tokenize_zh(text):
    words = jieba.lcut(text)
    return words

topic_model = BERTopic(language="multilingual", calculate_probabilities = True, embedding_model = model, verbose = True)

topics, probs = topic_model.fit_transform(my_input, embeddings = embeddings)

topic_model.get_topic_info()

# 3. VISUALIZATION OUTPUT

test1 = topic_model.visualize_topics()
test1.write_html("test1.html")

from html2image import Html2Image
hti = Html2Image()

hti.screenshot(
    html_file='test1.html', 
    save_as='topics.jpg',
    size=(650, 650)
)

test2 = topic_model.visualize_distribution(probs[350], min_probability=0.002)
test2.write_html("test2.html")
hti.screenshot(
    html_file='test2.html', 
    save_as='distribution.jpg',
    size=(800, 600)
)

test3 = topic_model.visualize_hierarchy(top_n_topics=25)
test3.write_html("test3.html")
hti.screenshot(
    html_file='test3.html', 
    save_as='hierarchy.jpg',
    size=(1000, 580)
)

test4 = topic_model.visualize_barchart(top_n_topics=8)
test4.write_html("test4.html")
hti.screenshot(
    html_file='test4.html', 
    save_as='barchart.jpg',
    size=(1000, 500)
)

test5 = topic_model.visualize_heatmap(n_clusters=20, width=750, height=750)
test5.write_html("test5.html")
hti.screenshot(
    html_file='test5.html', 
    save_as='heatmap.jpg',
    size=(750, 750)
)