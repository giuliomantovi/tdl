import pandas as pd
import pickle
import joblib
import gensim
from gensim.corpora import Dictionary
from time import time
import en_core_web_sm
import numpy as np
import spacy as spacy
from gensim.models.phrases import Phraser, Phrases
from gensim.utils import simple_preprocess


def sort_tuples(tuples, key_idx):
    # Step 1: Define a key function that returns the desired element of each tuple.
    key_func = lambda x: x[key_idx]

    # Step 2: Use the `sorted` function to sort the list of tuples using the key function.
    sorted_tuples = sorted(tuples, key=key_func)

    return sorted_tuples


def print_pickle(file):
    topic_model = joblib.load("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/topics/lda_model_16.jl")
    print('Topics:', '\n')
    print(sort_tuples(topic_model.show_topics(0), 0))
    #print(topic_model.get_topics())

    """obj = pd.read_pickle(file)
    print(obj['ObjectId(5714deed25ac0d8aee57e541)'])
    obj = pd.read_pickle("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/id_to_summary_lines.pickle")
    print(obj['5714deed25ac0d8aee57e541'])
    obj = pd.read_pickle("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/topics/dictionary.pickle")
    print(obj[600])

    ### Show a summary ###
    summaries = pd.read_pickle("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/id_to_summary_lines.pickle")
    #song_id = random_key_from_dict(summaries, seed=12)
    print('\n'.join(summaries['5714deed25ac0d8aee57e541']))"""


def evaluate_text(file):
    ### Compute topic distribution for unseen texts ###
    topic_model = joblib.load("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/topics/lda_model_16.jl")
    dictionary = pd.read_pickle("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/topics/dictionary.pickle")
    text = "The scars of your love remind me of us, they keep me thinking that we almost had it all"
    corpus = [text]
    corpus = complex_preprocess(corpus)
    dictionary = Dictionary(corpus)
    # dictionary.filter_extremes()   ### using this will filter out all words if your corpus is very small like here
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]
    for text in corpus_bow:
        print('\n', topic_model[text])


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


def lemmatization(spacy_nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = spacy_nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def complex_preprocess(corpus):
    t = time()
    unigrams = list(map(lambda text: simple_preprocess(text, min_len=1, max_len=100), corpus))
    print('Extracted', len(set(flatten_list(unigrams))), 'unigrams:', time() - t, '\t', unigrams[0][:10])
    bigram_model = Phraser(Phrases(unigrams))
    unigrams_bigrams = [bigram_model[text] for text in unigrams]
    del unigrams
    print('Extracted', len(set(flatten_list(unigrams_bigrams))), 'uni/bigrams:', time() - t, '\t',
          [b for b in unigrams_bigrams[0] if '_' in b][:10])
    spacy_nlp = en_core_web_sm.load()
    # spacy_nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    lemmatized_tokens = lemmatization(spacy_nlp, unigrams_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    del spacy_nlp
    del unigrams_bigrams
    print('Extracted', len(set(flatten_list(lemmatized_tokens))), 'lemmas:', time() - t, '\t', lemmatized_tokens[0])
    return lemmatized_tokens
