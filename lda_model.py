from time import time

import pandas as pd
import os
import re
from pprint import pprint
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import nltk
import gensim.corpora as corpora
from nltk.corpus import stopwords
import pyLDAvis.gensim
import pickle
import pyLDAvis
import string


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


nltk.download('stopwords')
stop_words = stopwords.words('english')


def remove_stopwords(texts):
    # stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


import re

WORD = re.compile(r'\w+')


def regTokenize(text):
    words = WORD.findall(text)
    return words


from collections import Counter

stopwords_dict = Counter(stop_words)


def preprocess_text(text):
    # tokens = word_tokenize(text)

    tokens = regTokenize(text)

    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopwords_dict]
    # print(filtered_tokens)
    """
    # stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(filtered_tokens)"""
    # print(text)
    return filtered_tokens


def create_model_chunks():
    cont = 0
    data_words = []
    languages = []
    song_index = 0
    for chunk in pd.read_csv('Genius_song_lyrics/song_lyrics.csv',
                             engine='c', chunksize=10000, usecols=['lyrics', 'language']):
        if cont == 50:
            break
        print(chunk)
        t = time()
        cont += 1
        language = chunk['language']
        indexes = []
        #removing all song texts that aren't english
        for i in range(song_index, song_index + len(chunk)):
            if language[i] != 'en':
                indexes.append(i - song_index)
        song_index += len(chunk)
        chunk = chunk.drop(index=chunk.index[indexes])

        lyrics = chunk['lyrics']
        language = chunk['language']
        # create_wordcloud(songs)
        languages += [x for x in language]
        data_words += [preprocess_text(text) for text in lyrics]

    # Create Dictionary
    #print(languages)
    id2word = corpora.Dictionary(data_words)
    id2word.filter_extremes()
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    num_topics = 5
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    #lda_model.save(fname="Genius_song_lyrics/lda_model_chunks/lda_mod")

    # visualize_topics(lda_model, num_topics, corpus, id2word)


def create_model():
    songs = pd.read_csv('Genius_song_lyrics/song_lyrics.csv', nrows=20)

    # Remove punctuation
    songs['lyrics_processed'] = \
        songs['lyrics'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    songs['lyrics_processed'] = \
        songs['lyrics_processed'].map(lambda x: x.lower())

    # Print out the first rows of papers
    # songs['lyrics_processed'].head()

    # create_wordcloud(songs)

    data = songs.lyrics_processed.values.tolist()  # ['song1','song2','song3']

    data_words = list(sent_to_words(data))
    # print(data_words) #[['token1','token2],['token1','token2']]
    # print(data_words[0])
    # remove stop words
    data_words = remove_stopwords(data_words)  # uguale a prima
    print(data_words)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    # print(corpus)
    # print(corpus[1])

    # number of topics
    num_topics = 10
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics)
    # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())
    # lda_model.save(fname="Genius_song_lyrics/lda_model/lda_mod")

    # visualize_topics(lda_model, num_topics, corpus, id2word)


def print_topics():
    lda_model = gensim.models.ldamodel.LdaModel.load("Genius_song_lyrics/lda_model/lda_mod")
    pprint(lda_model.print_topics())


def predict_text():
    lda_model = gensim.models.ldamodel.LdaModel.load("Genius_song_lyrics/lda_model/lda_mod")
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
    data_words = list(sent_to_words(song_text))
    # remove stop words
    data_words = remove_stopwords(data_words)
    id2word = corpora.Dictionary(data_words)
    corpus = [id2word.doc2bow(text) for text in data_words]

    print('\n', lda_model[corpus][0])


def create_wordcloud(songs):
    # Join the different processed titles together.
    long_string = ','.join(songs['lyrics_processed'].to_list())
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()


def visualize_topics(lda_model, num_topics, corpus, id2word):
    # Visualize the topics
    pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_' + str(num_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_' + str(num_topics) + '.html')
    LDAvis_prepared


# lda model con 50000 canzoni:
"""[(0,
  '0.010*"life" + 0.009*"never" + 0.008*"love" + 0.006*"hook" + 0.005*"say" + '
  '0.005*"way" + 0.005*"day" + 0.005*"could" + 0.004*"still" + 0.004*"would"'),
 (1,
  '0.008*"yo" + 0.004*"ya" + 0.004*"rock" + 0.004*"rap" + 0.004*"em" + '
  '0.003*"black" + 0.003*"yeah" + 0.003*"mic" + 0.003*"hook" + 0.003*"new"'),
 (2,
  '0.027*"la" + 0.021*"ich" + 0.019*"de" + 0.016*"und" + 0.015*"j" + '
  '0.014*"du" + 0.013*"les" + 0.012*"le" + 0.012*"die" + 0.011*"est"'),
 (3,
  '0.019*"yeah" + 0.015*"baby" + 0.014*"oh" + 0.013*"girl" + 0.010*"love" + '
  '0.010*"wanna" + 0.009*"uh" + 0.009*"ya" + 0.008*"want" + 0.008*"chorus"'),
 (4,
  '0.024*"nigga" + 0.020*"niggas" + 0.013*"fuck" + 0.011*"bitch" + 0.010*"ya" '
  '+ 0.008*"em" + 0.007*"ass" + 0.005*"yo" + 0.005*"money" + 0.005*"hook"')]"""

"""500 000 con 5 argomenti da portatile
[(0,
  '0.007*"people" + 0.005*"new" + 0.004*"one" + 0.003*"music" + 0.003*"big" + '
  '0.003*"think" + 0.003*"1" + 0.003*"2" + 0.003*"right" + 0.003*"going"'),
 (1,
  '0.005*"one" + 0.005*"shall" + 0.005*"may" + 0.004*"would" + 0.004*"upon" + '
  '0.004*"god" + 0.003*"must" + 0.003*"us" + 0.003*"power" + 0.003*"men"'),
 (2,
  '0.030*"love" + 0.023*"oh" + 0.018*"chorus" + 0.015*"baby" + 0.013*"yeah" + '
  '0.013*"let" + 0.013*"go" + 0.012*"got" + 0.012*"want" + 0.011*"time"'),
 (3,
  '0.021*"got" + 0.012*"shit" + 0.011*"fuck" + 0.010*"nigga" + 0.008*"em" + '
  '0.008*"bitch" + 0.008*"man" + 0.007*"niggas" + 0.007*"cause" + '
  '0.007*"money"'),
 (4,
  '0.009*"one" + 0.007*"life" + 0.007*"never" + 0.007*"see" + 0.007*"said" + '
  '0.006*"time" + 0.005*"could" + 0.005*"would" + 0.005*"away" + 0.005*"day"')]"""