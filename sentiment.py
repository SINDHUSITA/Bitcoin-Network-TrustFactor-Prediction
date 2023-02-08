import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from re import sub

network_new = pd.read_csv("network.csv")
network = pd.read_csv("network.csv")

network = network[["id","notes"]]
network = network.dropna()
network.drop_duplicates(inplace = True)
network.reset_index(inplace=True, drop = True)

def clean_text(text):
    # Clean the text
    text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{2,}", " ", text)
    text = sub(r"won\'t", "will not", text)
    text = sub(r"can\'t", "can not", text)
    text = sub(r"n\'t", " not", text)
    # to join not/no into the next word
    text = sub("not ", " NOT", text)
    text = sub("no ", " NO", text)
    return text

def remove_stopwords(tokenized_text):
    text = [word for word in tokenized_text if word not in nltk.corpus.stopwords.words('english')]
    return text

def text_to_word_list(text):
    text = str(text)
    text = text.lower()
    text = clean_text(text)
    text = text.split()

    return text if len(text) >0 else ""

network['orignal_notes'] = network['notes']
network.loc[:,'notes'] = network.loc[:,'notes'].apply(lambda x: text_to_word_list(x))
network_new = network.loc[network['notes'] != ""]
network.reset_index(inplace=True, drop = True)

#stopwords
network.loc[:,'notes'] = network.loc[:,'notes'].apply(lambda x: remove_stopwords(x))

network["notes"].shape

#Load bert model
import spacy

nlp = spacy.load("en_core_web_sm")

# Utility function for generating sentence embedding from the text
def get_embeddinngs(text):
    return nlp(" ".join(text)).vector

# Generating sentence embedding from the text
network['emb'] = network['notes'].apply(get_embeddinngs)

def del_emp(a):
  return len(a) == 0

network.head()

network = network[network['emb'].map(lambda d: len(d)) > 0]

## Import libraries
from nltk.cluster import KMeansClusterer
import nltk

def clustering_question(data):
    X = np.array(data['emb'].tolist())

    kclusterer = KMeansClusterer(2, distance=nltk.cluster.util.cosine_distance,repeats=25,avoid_empty_clusters=True)

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

    data['cluster'] = pd.Series(assigned_clusters, index=data.index)
    data['centroid'] = data['cluster'].apply(lambda x: kclusterer.means()[x])

    return data, assigned_clusters

data, assigned_clusters = clustering_question(network)

data.cluster.value_counts()

updated_data = data[['id','cluster']]
updated_data['cluster'].replace(0,-1,inplace=True)
no_notes_id = list(set(network_new['id'])-set(updated_data['id']))
sentiment_0 = np.zeros(len(no_notes_id))
d = {'id': no_notes_id, 'cluster': sentiment_0}
df = pd.DataFrame(data=d)
all_edges = pd.concat([updated_data, df], axis=0)
all_edges.to_csv('sentiment.csv',index = False)