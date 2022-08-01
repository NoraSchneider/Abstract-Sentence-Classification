import pandas as pd
import numpy as np
import re, string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

from sklearn.utils import resample
from gensim.models import Word2Vec, KeyedVectors


# STOPWORD REMOVAL
def stopword(string):
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Lemmatization
def lemmatizer(text):
    # Initialize the lemmatizer
    wl = WordNetLemmatizer()
    word_pos_tags = nltk.pos_tag(word_tokenize(text))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(a)


# Stemming
def stemmer(text):
    s1 = SnowballStemmer('english')
    words = word_tokenize(text)
    a = [s1.stem(word) for word in words]
    return " ".join(a)


def preprocess_text(text: str, mode: str = "", remove_numplaceholder: bool = True):
    # convert to lowercase, strip and remove punctuations
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)

    if remove_numplaceholder:
        helper = string.punctuation
    else:
        helper = string.punctuation.replace("@", "")
    text = re.compile('[%s]' % re.escape(helper)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = stopword(text)

    if mode == "lemmatization":
        text = lemmatizer(text)
    elif mode == "stemming":
        text = stemmer(text)

    return text


def balance_data(data: pd.DataFrame(), class_mapping: dict):
    balanced_data = pd.DataFrame()

    for label in class_mapping.keys():
        size, replace = class_mapping[label]
        resampled_data = [resample(data.loc[data['label'] == int(label)],
                                   replace=replace,
                                   n_samples=size,
                                   random_state=321)]
        balanced_data = pd.concat([balanced_data] + resampled_data)
    return balanced_data


def get_neighbour_idx(df: pd.DataFrame()):

    df = df.sort_values('abstract_id',ascending=True).reset_index(drop=True)

    df["prev_idx"] = 0
    df["next_idx"] = 0
    max_sent_len = len(df["idx"][0])

    for i in range(len(dev_data)):

        if df["line_relative"][i] == 0:
            df["prev_idx"][i] = np.zeros( shape=( max_sent_len ))
            df["next_idx"][i] = df["idx"][i+1]

        elif df["line_relative"][i] == 1:
            df["prev_idx"][i] = df["idx"][i-1]
            df["next_idx"][i] = np.zeros( shape=( max_sent_len ))
        else:
            df["prev_idx"][i] = df["idx"][i-1]
            df["next_idx"][i] = df["idx"][i+1] 


def get_diffed_vecs(df: pd.DataFrame()):

    df.sort_values(['abstract_id', "line_relative"], inplace=True, ascending=[True, True]).reset_index(drop=True)
    
    max_sent_len = len(df['tokens'][0])
    diffed_vecs =  np.zeros( shape=( len(df["avg_vectors"]), len(df["avg_vectors"][0]) ))

    for i in range( len(df["avg_vectors"])):

        if df["line_relative"][i] == 0:
            diffed_vecs[i] = df["avg_vectors"][i] - df["avg_vectors"][i+1]

        elif df["line_relative"][i] == 1:
            diffed_vecs[i] = df["avg_vectors"][i] - df["avg_vectors"][i-1]
        else:
            diffed_vecs[i] = df["avg_vectors"][i]*2 - df["avg_vectors"][i-1] - df["avg_vectors"][i+1]   
            
    return diffed_vecs, df['label'].values


def words_to_idx(df: pd.DataFrame(), kv_model, max_sent_len: int = 150,
                 pad_val: int = 0, save_name: str =""):

    idx_dict = kv_model.key_to_index
        
    def convert_idx_list(sent):
        idx_arr = np.full( shape=( max_sent_len), fill_value=pad_val)
        
        idx= np.array( [ idx_dict[word]+1 for word in sent if word in idx_dict] )
        
        if not len(idx): return np.NaN
    
        idx_arr[0: min(idx.shape[0], max_sent_len) ] = idx[0: min(idx.shape[0], max_sent_len)]
        
        return idx_arr
        
    
    df["idx"] = df['tokens'].apply(lambda x: convert_idx_list(x))
    
    df = df.dropna()
    
    if save_name:
        df.to_pickle( f"{save_name}.pkl" )
        
    return df



def vectorize_dataset(df: pd.DataFrame(), kv_model, save_name: str=""):
    
    def get_avg_vector(token_sentence):
        
        word_embeds = np.array( [ kv_model[x] for x in token_sentence if x in vocab] )   
        if word_embeds.size:
            return word_embeds.mean(axis=0)
        else: return np.array([])
        
    #get vocabulary of the Word2Vec model
    vocab = kv_model.key_to_index
    df["avg_vectors"] = df["tokens"].apply(lambda x: get_avg_vector(x))
    
    if save_name:
        df.to_pickle( f"{save_name}.pkl" )
        
    return df