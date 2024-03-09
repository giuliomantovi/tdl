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


topics_dict = {0: ""}

def sort_tuples(tuples, key_idx):
    # Step 1: Define a key function that returns the desired element of each tuple.
    key_func = lambda x: x[key_idx]

    # Step 2: Use the `sorted` function to sort the list of tuples using the key function.
    sorted_tuples = sorted(tuples, key=key_func)

    return sorted_tuples


def print_pickle(file):
    topic_model = joblib.load("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/topics/lda_model_16.jl")
    print('Topics:', '\n')
    for tup in sort_tuples(topic_model.show_topics(0), 0):
        print(tup)
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

SOME_FREE_SONGTEXT = "So one night I said to Kathy We gotta get away somehow Go somewhere south and somewhere warm But for God's sake let's go now. And Kathy she sort of looks at me And asks where I wanna go So I look back and I hear me say I don't care but we gotta go chorus and key change And all the other people Who slepwalk thru their days Just sort of faded out of sight When we two drove away And ev'ry day we travelled We were lookin' to get wise And we learned what was the truth And we learned what were the lies And in LA we bought a bus Sort of old and not too smart So for six hundred and fifty bucks We got out and made a start We hit the road down to the South And drove into Mexico That old bus was some old wreck But it just kept us on the road. chorus etc We drove up to Alabam And a farmer gave us some jobs We worked them crops all night and day And at night we slept like dogs We got paid and Kathy said to me It's time to make a move again And when I looked into her eyes I saw more than a friend. chorus etc And now we've stopped our travels And we sold the bus in Texas And we made our home in Austin And for sure it ain't no palace And Kathy and me we settled down And now our first kid's on the way Kathy and me and that old bus We did real good to get away."


def evaluate_text(file):
    ### Compute topic distribution for unseen texts ###
    topic_model = joblib.load("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/topics/lda_model_16.jl")
    dictionary = pd.read_pickle("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/topics/dictionary.pickle")
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
    dictionary = Dictionary(corpus)
    # dictionary.filter_extremes()   ### using this will filter out all words if your corpus is very small like here
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]
    print(corpus_bow)
    for text in corpus_bow:
        print('\n', topic_model[text])
    print(dictionary[0])


#IDEA: FARE MAPPA CON KEY=PAROLA E VALUE=TOPIC, PER OGNI LEMMA NEL CORPUS BOW INCREMENTO UN CONTATORE PER TOPIC
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
