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


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(texts):
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    # stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def create_model():

    songs = pd.read_csv('Genius_song_lyrics/song_lyrics.csv', nrows=10000)

    # Remove punctuation
    songs['lyrics_processed'] = \
        songs['lyrics'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    songs['lyrics_processed'] = \
        songs['lyrics_processed'].map(lambda x: x.lower())
    # Print out the first rows of papers
    songs['lyrics_processed'].head()

    create_wordcloud(songs)

    data = songs.lyrics_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    print(data_words[:1][0][:30])

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    print(corpus[:1][0][:30])

    # number of topics
    num_topics = 10
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    lda_model.save(fname="Genius_song_lyrics/lda_model/lda_mod")

    # visualize_topics(lda_model, num_topics, corpus, id2word)


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
