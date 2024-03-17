from time import time
import en_core_web_sm
from gensim.models.phrases import Phraser, Phrases
from lyrics_classification.text_processing import *


def evaluate_text(file):
    ### Compute topic distribution for unseen texts using WASABI LDA_MODEL AND DICTIONARY###
    topic_model = joblib.load("../WASABI_DB/topics/lda_model_16.jl")
    dictionary = pd.read_pickle("/WASABI_DB/topics/dictionary.pickle")
    song_text = """When the rain is blowing in your face
        And the whole world is on your case
        I could offer you a warm embrace
        To make you feel my love
        When the evening shadows and the stars appear
        And there is no one there to dry your tears
        I could hold you for a million years
        To make you feel my love
        I know you havent made your mind up yet
        But I will never do you wrong
        Ive known it from the moment that we met
        No doubt in my mind where you belong
        Id go hungry, Id go black and blue
        Id go crawling down the avenue
        No, theres nothing that I wouldnt do
        To make you feel my love
        The storms are raging on the rolling sea
        And on the highway of regret
        The winds of change are blowing wild and free
        You aint seen nothing like me yet
        I could make you happy, make your dreams come true
        Nothing that I wouldnt do
        Go to the ends of the Earth for you
        To make you feel my love
        To make you feel my love"""
    corpus = [song_text]
    corpus = complex_preprocess(corpus)
    #dictionary = Dictionary(corpus)
    # dictionary.filter_extremes()   ### using this will filter out all words if your corpus is very small like here
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]
    print(corpus_bow)
    for text in corpus_bow:
        print('\n', topic_model[text])


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




