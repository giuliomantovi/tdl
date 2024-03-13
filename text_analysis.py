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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download("punkt")
nltk.download("stopwords")

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
    # print(topic_model.get_topics())

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


# IDEA: FARE MAPPA CON KEY=PAROLA E VALUE=TOPIC, PER OGNI LEMMA NEL CORPUS BOW INCREMENTO UN CONTATORE PER TOPIC
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


import re

WORD = re.compile(r'\w+')


def regTokenize(text):
    words = WORD.findall(text)
    return words


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
from collections import Counter
stopwords_dict = Counter(stop_words)
def preprocess_classifier_text(text):
    # tokens = word_tokenize(text)
    tokens = regTokenize(text)

    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopwords_dict]

    #stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(filtered_tokens)


def create_text_classifier():
    # Genre inference from genius lyrics and genre with
    cont = 0
    x_preprocessed = []
    y_preprocessed = []
    y=any
    for chunk in pd.read_csv('Genius_song_lyrics/song_lyrics.csv',
                             engine='c', chunksize=100000, usecols=['lyrics', 'tag']):
        if cont == 50:
            break
        print(chunk)
        cont += 1
        x, y = chunk.lyrics, chunk.tag
        y_preprocessed += [genre for genre in y]
        x_preprocessed += [preprocess_classifier_text(text) for text in x]
    #print(x_preprocessed)

    vectorizer = TfidfVectorizer()
    x_transformed = vectorizer.fit_transform(x_preprocessed)
    classifier = MultinomialNB()
    classifier.fit(x_transformed, y_preprocessed)

    #save the model to disk (previous one trained on first 2 million songs, 0.46% accuracy
    classifier_filename = 'Genius_song_lyrics/genre_classifier.sav'
    pickle.dump(classifier, open(classifier_filename, 'wb'))

    vectorizer_filename = 'Genius_song_lyrics/vectorizer.pk'
    pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
    #evaluate_text_classifier()


def evaluate_text_classifier():
    loaded_classifier = pickle.load(open("Genius_song_lyrics/genre_classifier.sav", 'rb'))
    loaded_vectorizer = pickle.load(open("Genius_song_lyrics/vectorizer.pk", 'rb'))

    cont = 0
    x_preprocessed = []
    y_preprocessed = []
    y = any
    for chunk in pd.read_csv('Genius_song_lyrics/song_lyrics.csv', engine='c', chunksize=100000):
        if cont == 20:
            #print(chunk)
            x, y = chunk.lyrics, chunk.tag
            y_preprocessed += [genre for genre in y]
            x_preprocessed += [preprocess_classifier_text(text) for text in x]
            break
        cont += 1

    #test_data = pd.read_csv('Genius_song_lyrics/song_lyrics.csv', engine='c')
    #x_test, y_test = test_data.lyrics[2000000:2001000], test_data.tag[2000000:2001000]

    x_test_transformed = loaded_vectorizer.transform(x_preprocessed)

    y_pred = loaded_classifier.predict(x_test_transformed)

    accuracy = accuracy_score(y_preprocessed, y_pred)
    print("Accuracy:", accuracy)


def predict_genre_from_lyrics():
    # load the model from disk
    loaded_classifier = pickle.load(open("Genius_song_lyrics/genre_classifier.sav", 'rb'))
    loaded_vectorizer = pickle.load(open("Genius_song_lyrics/vectorizer.pk", 'rb'))

    X_test = """Buddy, you're a boy, make a big noise
            Playing in the street, gonna be a big man someday
            You got mud on your face, you big disgrace
            Kicking your can all over the place, singin'
            We will, we will rock you
            We will, we will rock you
            Buddy, you're a young man, hard man
            Shouting in the street, gonna take on the world someday
            You got blood on your face, you big disgrace
            Waving your banner all over the place
            We will, we will rock you, sing it
            We will, we will rock you
            Buddy, you're an old man, poor man
            Pleading with your eyes, gonna make you some peace someday
            You got mud on your face, big disgrace
            Somebody better put you back into your place
            We will, we will rock you, sing it
            We will, we will rock you, everybody
            We will, we will rock you, hmm
            We will, we will rock you
            Alright"""

    X_test_preprocessed = [preprocess_classifier_text(text) for text in X_test]
    X_test_transformed = loaded_vectorizer.transform(X_test_preprocessed)
    #print(X_test_transformed)
    result = loaded_classifier.predict(X_test_transformed)
    print(result)

    """summaries = pd.read_pickle("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/topics/song_id_to_topics.pickle")
    print(summaries["ObjectId(5714deed25ac0d8aee57e542)"])
    obj = pd.read_pickle("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/id_to_summary_lines.pickle")
    print(obj["5714deed25ac0d8aee57e542"])"""
    """[(4, 0.05279845), (8, 0.06588726), (13, 0.38976097), (16, 0.022661319), (18, 0.022483176), (27, 0.010201428), (30, 0.054924987), (32, 0.016545713), (40, 0.010497759), (41, 0.04243385), (44, 0.020272728), (47, 0.010643412), (48, 0.013438855), (51, 0.025851179), (52, 0.013056019), (55, 0.0149408085), (56, 0.010882135), (58, 0.017950308)]
["it ain't nothing wrong but y'all know me", "i'm a loc and my loc niggaz need me bitch", "when i'm in the club i don't see real niggaz", "i see a bunch of wannabe's in it"]"""

    """(0, '0.426*"never" + 0.072*"wish" + 0.061*"miss" + 0.052*"know" + 0.050*"really" + 0.040*"ve" + 0.029*"have" + 0.024*"guess" + 0.018*"mean" + 0.011*"hurt"')
(1, '0.339*"feel" + 0.163*"fall" + 0.078*"same" + 0.054*"real" + 0.045*"feeling" + 0.026*"inside" + 0.025*"when" + 0.023*"moment" + 0.015*"seem" + 0.014*"see"')
(2, '0.200*"too" + 0.133*"stop" + 0.128*"much" + 0.104*"talk" + 0.043*"house" + 0.037*"listen" + 0.031*"when" + 0.018*"just" + 0.016*"know" + 0.012*"wheel"')
(3, '0.218*"man" + 0.047*"woman" + 0.034*"chorus" + 0.028*"lady" + 0.016*"verse" + 0.015*"u" + 0.013*"dem" + 0.006*"mi" + 0.006*"e" + 0.005*"yuh"')
(4, '0.248*"could" + 0.246*"world" + 0.124*"change" + 0.078*"remember" + 0.036*"whole" + 0.024*"see" + 0.023*"when" + 0.010*"have" + 0.008*"find" + 0.006*"ugly"')
(5, '0.294*"heart" + 0.159*"break" + 0.064*"beat" + 0.030*"part" + 0.024*"apart" + 0.021*"broken" + 0.018*"make" + 0.017*"start" + 0.014*"weak" + 0.014*"heat"')
(6, '0.275*"there" + 0.185*"here" + 0.135*"where" + 0.104*"call" + 0.078*"something" + 0.061*"hear" + 0.020*"out" + 0.012*"somewhere" + 0.012*"everywhere" + 0.009*"know"')
(7, '0.243*"away" + 0.110*"light" + 0.077*"free" + 0.075*"stay" + 0.067*"far" + 0.067*"fly" + 0.064*"sky" + 0.041*"set" + 0.035*"fade" + 0.019*"wing"')
(8, '0.260*"leave" + 0.183*"nothing" + 0.072*"must" + 0.052*"wrong" + 0.047*"reason" + 0.037*"ve" + 0.027*"else" + 0.020*"there" + 0.016*"mistake" + 0.015*"behind"')
(9, '0.453*"come" + 0.253*"back" + 0.164*"keep" + 0.035*"bring" + 0.010*"when" + 0.008*"track" + 0.003*"pressure" + 0.003*"take" + 0.003*"along" + 0.003*"see"')
(10, '0.405*"now" + 0.206*"right" + 0.137*"all" + 0.036*"water" + 0.023*"at" + 0.022*"wild" + 0.019*"wrong" + 0.009*"start" + 0.007*"left" + 0.007*"make"')
(11, '0.216*"turn" + 0.095*"around" + 0.054*"round" + 0.039*"when" + 0.038*"low" + 0.025*"warm" + 0.022*"snow" + 0.021*"spin" + 0.017*"see" + 0.017*"circle"')
(12, '0.054*"have" + 0.046*"over" + 0.043*"laugh" + 0.033*"soon" + 0.033*"fun" + 0.031*"pick" + 0.031*"piece" + 0.028*"yet" + 0.023*"hole" + 0.022*"then"')
(13, '0.022*"get" + 0.022*"nigga" + 0.022*"fuck" + 0.021*"shit" + 0.017*"bitch" + 0.014*"ain" + 0.010*"ass" + 0.009*"yo" + 0.008*"niggas" + 0.007*"hit"')
(14, '0.120*"may" + 0.048*"rise" + 0.046*"wind" + 0.036*"blow" + 0.028*"build" + 0.028*"past" + 0.026*"bone" + 0.026*"stone" + 0.026*"future" + 0.022*"reach"')
(15, '0.176*"night" + 0.146*"more" + 0.123*"dream" + 0.061*"sun" + 0.053*"star" + 0.040*"no" + 0.031*"sleep" + 0.027*"moon" + 0.026*"dark" + 0.018*"shadow"')
(16, '0.380*"want" + 0.130*"ain" + 0.078*"bad" + 0.057*"rock" + 0.051*"roll" + 0.035*"somebody" + 0.035*"nobody" + 0.027*"just" + 0.019*"really" + 0.018*"know"')
(17, '0.154*"walk" + 0.065*"line" + 0.061*"street" + 0.060*"door" + 0.046*"step" + 0.043*"see" + 0.042*"just" + 0.040*"when" + 0.021*"pretend" + 0.021*"know"')
(18, '0.056*"god" + 0.046*"lord" + 0.044*"name" + 0.043*"soul" + 0.025*"pray" + 0.021*"earth" + 0.021*"great" + 0.020*"son" + 0.020*"peace" + 0.017*"faith"')
(19, '0.274*"have" + 0.084*"hand" + 0.071*"forget" + 0.054*"today" + 0.049*"when" + 0.048*"young" + 0.044*"shake" + 0.024*"dear" + 0.018*"just" + 0.016*"spend"')
(20, '0.043*"face" + 0.036*"air" + 0.033*"cut" + 0.032*"pass" + 0.030*"breathe" + 0.029*"scream" + 0.028*"pull" + 0.026*"hell" + 0.021*"devil" + 0.019*"read"')
(21, '0.178*"one" + 0.152*"ever" + 0.151*"wait" + 0.102*"last" + 0.073*"forever" + 0.026*"blame" + 0.026*"steal" + 0.017*"minute" + 0.014*"know" + 0.013*"letter"')
(22, '0.151*"thing" + 0.134*"little" + 0.106*"well" + 0.095*"make" + 0.063*"good" + 0.052*"big" + 0.038*"just" + 0.025*"sad" + 0.019*"try" + 0.014*"while"')
(23, '0.080*"play" + 0.075*"move" + 0.069*"song" + 0.064*"dance" + 0.050*"sing" + 0.042*"everybody" + 0.035*"music" + 0.033*"sound" + 0.027*"party" + 0.017*"floor"')
(24, '0.039*"child" + 0.030*"bear" + 0.027*"land" + 0.025*"sea" + 0.022*"mother" + 0.021*"angel" + 0.020*"red" + 0.019*"tree" + 0.014*"mountain" + 0.011*"hill"')
(25, '0.496*"time" + 0.125*"mind" + 0.048*"first" + 0.036*"waste" + 0.019*"sign" + 0.018*"gun" + 0.017*"second" + 0.014*"o" + 0.013*"next" + 0.011*"kill"')
(26, '0.175*"home" + 0.129*"alone" + 0.082*"own" + 0.071*"someone" + 0.064*"save" + 0.033*"tired" + 0.030*"sick" + 0.023*"scar" + 0.021*"shame" + 0.020*"take"')
(27, '0.254*"live" + 0.064*"wake" + 0.063*"easy" + 0.059*"city" + 0.048*"tomorrow" + 0.036*"river" + 0.027*"memory" + 0.023*"yesterday" + 0.021*"find" + 0.018*"sorrow"')
(28, '0.361*"day" + 0.089*"new" + 0.071*"body" + 0.050*"follow" + 0.049*"happy" + 0.042*"make" + 0.021*"count" + 0.019*"hour" + 0.019*"someday" + 0.018*"everyday"')
(29, '0.248*"life" + 0.128*"lose" + 0.122*"lie" + 0.107*"everything" + 0.040*"truth" + 0.036*"anything" + 0.029*"happen" + 0.027*"carry" + 0.018*"choose" + 0.009*"tell"')
(30, '0.369*"don" + 0.324*"t" + 0.092*"win" + 0.035*"care" + 0.030*"know" + 0.026*"didn" + 0.014*"anymore" + 0.014*"doesn" + 0.010*"matter" + 0.010*"worry"')
(31, '0.401*"will" + 0.090*"die" + 0.073*"cry" + 0.060*"burn" + 0.053*"tear" + 0.052*"fire" + 0.045*"pain" + 0.025*"fear" + 0.013*"flame" + 0.012*"bleed"')
(32, '0.678*"get" + 0.107*"run" + 0.034*"better" + 0.029*"enough" + 0.017*"ve" + 0.016*"on" + 0.013*"got" + 0.007*"lucky" + 0.007*"problem" + 0.006*"nowhere"')
(33, '0.196*"eye" + 0.146*"always" + 0.091*"watch" + 0.061*"close" + 0.061*"together" + 0.060*"open" + 0.047*"smile" + 0.036*"see" + 0.019*"slip" + 0.017*"safe"')
(34, '0.091*"true" + 0.085*"blue" + 0.078*"year" + 0.048*"white" + 0.040*"send" + 0.040*"everyone" + 0.034*"brother" + 0.028*"ring" + 0.024*"strange" + 0.019*"christmas"')
(35, '0.685*"re" + 0.043*"when" + 0.033*"know" + 0.025*"trouble" + 0.019*"number" + 0.018*"cool" + 0.017*"cause" + 0.014*"just" + 0.005*"tell" + 0.005*"make"')
(36, '0.637*"love" + 0.130*"give" + 0.025*"lover" + 0.022*"touch" + 0.013*"hurt" + 0.013*"know" + 0.010*"more_than" + 0.008*"share" + 0.008*"when" + 0.008*"darle"')
(37, '0.101*"old" + 0.046*"town" + 0.036*"catch" + 0.022*"dog" + 0.020*"bird" + 0.016*"mile" + 0.013*"glad" + 0.012*"hang" + 0.010*"swallow" + 0.010*"cat"')
(38, '0.501*"not" + 0.099*"word" + 0.038*"side" + 0.027*"afraid" + 0.025*"know" + 0.021*"belong" + 0.013*"anyone" + 0.011*"when" + 0.008*"ill" + 0.008*"try"')
(39, '0.024*"blood" + 0.023*"death" + 0.009*"evil" + 0.007*"flesh" + 0.006*"human" + 0.006*"force" + 0.006*"kill" + 0.006*"fear" + 0.006*"soul" + 0.006*"darkness"')
(40, '0.139*"down" + 0.080*"help" + 0.063*"ready" + 0.056*"ground" + 0.036*"up" + 0.035*"alright" + 0.029*"knee" + 0.026*"when" + 0.026*"hit" + 0.018*"take"')
(41, '0.639*"m" + 0.169*"i" + 0.036*"ride" + 0.019*"cause" + 0.018*"train" + 0.017*"sorry" + 0.005*"know" + 0.005*"just" + 0.004*"about" + 0.003*"when"')
(42, '0.201*"again" + 0.161*"would" + 0.110*"end" + 0.053*"once" + 0.047*"road" + 0.044*"learn" + 0.033*"start" + 0.024*"ve" + 0.021*"see" + 0.020*"begin"')
(43, '0.463*"let" + 0.118*"tonight" + 0.107*"show" + 0.035*"hot" + 0.031*"just" + 0.030*"make" + 0.025*"know" + 0.014*"flow" + 0.013*"take" + 0.009*"can"')
(44, '0.423*"do" + 0.169*"baby" + 0.112*"wanna" + 0.083*"why" + 0.049*"tell" + 0.032*"know" + 0.012*"babe" + 0.008*"really" + 0.006*"just" + 0.006*"make"')
(45, '0.574*"so" + 0.103*"long" + 0.037*"lonely" + 0.031*"beautiful" + 0.018*"just" + 0.011*"knock" + 0.010*"know" + 0.010*"proud" + 0.009*"beauty" + 0.008*"d_rather"')
(46, '0.338*"think" + 0.117*"friend" + 0.071*"black" + 0.069*"hate" + 0.034*"hope" + 0.033*"when" + 0.030*"summer" + 0.024*"see" + 0.017*"know" + 0.016*"winter"')
(47, '0.217*"only" + 0.205*"still" + 0.110*"have" + 0.050*"strong" + 0.049*"use" + 0.035*"promise" + 0.021*"see" + 0.019*"chain" + 0.012*"know" + 0.009*"make"')
(48, '0.467*"say" + 0.107*"believe" + 0.088*"boy" + 0.037*"just" + 0.035*"fine" + 0.027*"know" + 0.024*"trust" + 0.016*"tell" + 0.015*"goodbye" + 0.012*"make"')
(49, '0.130*"stand" + 0.107*"should" + 0.081*"cold" + 0.057*"sometimes" + 0.029*"search" + 0.021*"cover" + 0.018*"see" + 0.017*"forgive" + 0.015*"have" + 0.014*"answer"')
(50, '0.095*"head" + 0.077*"good" + 0.056*"dead" + 0.034*"drink" + 0.032*"alive" + 0.031*"room" + 0.028*"bed" + 0.023*"eat" + 0.014*"mouth" + 0.014*"kill"')
(51, '0.200*"how" + 0.076*"people" + 0.038*"many" + 0.034*"fool" + 0.023*"kid" + 0.020*"write" + 0.018*"very" + 0.017*"other" + 0.016*"thank" + 0.016*"different"')
(52, '0.183*"girl" + 0.071*"like" + 0.045*"kiss" + 0.033*"kind" + 0.030*"pretty" + 0.022*"meet" + 0.022*"wear" + 0.021*"guy" + 0.019*"hair" + 0.019*"just"')
(53, '0.241*"way" + 0.087*"hard" + 0.073*"just" + 0.059*"work" + 0.041*"money" + 0.035*"pay" + 0.025*"make" + 0.021*"morning" + 0.020*"wonder" + 0.018*"buy"')
(54, '0.079*"rain" + 0.061*"deep" + 0.057*"hide" + 0.056*"wall" + 0.039*"power" + 0.028*"perfect" + 0.028*"gold" + 0.027*"secret" + 0.025*"see" + 0.020*"wash"')
(55, '0.301*"need" + 0.096*"high" + 0.075*"sweet" + 0.031*"mama" + 0.029*"honey" + 0.028*"taste" + 0.026*"daddy" + 0.016*"good" + 0.016*"take" + 0.013*"make"')
(56, '0.445*"can" + 0.156*"take" + 0.032*"maybe" + 0.031*"crazy" + 0.029*"drive" + 0.021*"slow" + 0.016*"just" + 0.015*"car" + 0.014*"see" + 0.014*"make"')
(57, '0.268*"look" + 0.213*"know" + 0.070*"see" + 0.039*"shine" + 0.034*"mine" + 0.027*"just" + 0.024*"when" + 0.017*"dirty" + 0.015*"make" + 0.015*"na_na"')
(58, '0.030*"fight" + 0.024*"war" + 0.013*"sell" + 0.009*"control" + 0.008*"state" + 0.007*"law" + 0.007*"fucking" + 0.007*"machine" + 0.006*"soldier" + 0.006*"tv"')
(59, '0.191*"hold" + 0.159*"d" + 0.121*"place" + 0.055*"arm" + 0.041*"just" + 0.031*"tight" + 0.029*"when" + 0.028*"hand" + 0.028*"know" + 0.025*"wouldn"')"""
