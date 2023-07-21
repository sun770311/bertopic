# BERTopic for Chinese Language
Hannah Sun   
hysun@umich.edu  

## Project Mentor
Dr. Patrick Pang, Assistant Professor and PhD supervisor in the Faculty of Applied Sciences at Macao Polytechnic University.
<br />

## Goals
* Adapt Maarten Grootendorst's unsupervised machine learning model [BERTopic](https://github.com/MaartenGr/BERTopic) to process Chinese text input. Used in Natural Language Processing (NLP), BERTopic is "a topic modeling technique that leverages transformers and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions". 
* Generate topic, cluster hierarchy, and topic similarity visualizations from a set of 20,000+ patient evaluations retrieved from [HaoDF](https://www.haodf.com/), a leading online healthcare service provider in China.
<br />

## Challenges
BERTopic is originally designed for English, which naturally has spaces between words that the topic modeling technique uses to distinguish keywords for each topic. On the other hand, Chinese characters do not have spaces in between them and more than one character can be joined to create a phrae that translates to a singular English word. Therefore, we need to split Chinese words and phrases appropriately in the pre-processing steps.
<br />

## Required Packages and Files
Listed in requirements.txt
<br />
   
## two_functions.py 
The first function split_input() takes in 3 files- 
1. haodfcomments.csv with the original input text, 
2. fixes-zh.txt with the list of stopwords to remove,
3. split_comments.csv as the output csv with split words/phrases (can rename to file of your choice)
and pre-processes the input. We strip the stopwords and identify Chinese words and phrases through word segmentation. [Jieba](https://pypi.org/project/jieba/), a Python Chinese word segmentation module, is used to facilitate this step. The processed comments are written to split_comments.csv to be taken in by the second function.
<br />

The second function bert() has 2 parameters-
1. input_file: name of output file from split_input(), in this case split_comments.csv
2. path_name: the directory path to output files generated with BERTopic.
We flatten the input list of lists into a single list before initializing the topic model with some specifications. Next, we fit the model on this list of documents and generate topics. Finally, we output 5 visualizations- topic map, topic distribution, topic hierarchy, topic word scores, and topic similarity heatmap- to the user's directory path of choice.
<br />

## References
[BERTopic Github Repository](https://github.com/MaartenGr/BERTopic) by Maarten Grootendorst
<br />

[BERTopic API](https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.get_representative_docs) by Maarten Grootendorst
<br />

[Python- Chinese word segmentation and remove stop words to keep only Chinese characters](https://blog.csdn.net/lztttao/article/details/104723228) by lztttao
<br />

[NLP practical learning (2): Bertopic-based news topic modeling](https://blog.csdn.net/weixin_47113960/article/details/125373275) by 银河小铁骑plus
<br />
<br />  