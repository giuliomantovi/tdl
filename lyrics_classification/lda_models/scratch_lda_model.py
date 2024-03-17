import os
from pprint import pprint
from wordcloud import WordCloud
import gensim.corpora as corpora
import pyLDAvis.gensim
import pickle
import pyLDAvis
from lyrics_classification.text_processing import *


def preprocess_text(text):
    tokens = regTokenize(text)
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopwords_dict]
    # stemming not used for model creation since it requires too much time
    """stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(filtered_tokens)"""
    return filtered_tokens


def create_model_chunks():
    cont = 0
    data_words = []
    song_index = 0
    for chunk in pd.read_csv('../Genius_song_lyrics_DB/song_lyrics.csv',
                             engine='c', chunksize=100000, usecols=['lyrics', 'language']):
        if cont == 5:
            break
        print(chunk)
        cont += 1
        language = chunk['language']
        indexes = []
        # removing all song texts that aren't english
        for i in range(song_index, song_index + len(chunk)):
            if language[i] != 'en':
                indexes.append(i - song_index)
        song_index += len(chunk)
        chunk = chunk.drop(index=chunk.index[indexes])
        lyrics = chunk['lyrics']

        data_words += [preprocess_text(text) for text in lyrics]

    # create_wordcloud(songs)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    id2word.filter_extremes()
    # Create Corpus
    #texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_words]

    num_topics = 4
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    lda_model.save(fname="Genius_song_lyrics_DB/lda_model/lda_mod")

    # visualize_topics(lda_model, num_topics, corpus, id2word)


def print_topics():
    lda_model = gensim.models.ldamodel.LdaModel.load("Genius_song_lyrics_DB/lda_model/lda_mod")
    pprint(lda_model.print_topics())


def predict_text(song_text):
    lda_model = gensim.models.ldamodel.LdaModel.load("Genius_song_lyrics_DB/lda_model/lda_mod")
    song_text = """It's late in the evening
She's wondering what clothes to wear
She puts on her makeup
And brushes her long blonde hair
And then she asks me
"Do I look all right?"
And I say, "Yes, you look wonderful tonight"
"""
    data_words = [preprocess_text(song_text)]
    print(data_words)
    id2word = corpora.Dictionary.load("Genius_song_lyrics_DB/lda_model/lda_mod.id2word")
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
    wordcloud.to_image().save("Genius_song_lyrics_DB/lda_model/wordcloud.png")


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


"""MODELLO CORRENTE: 500 000 con 4 argomenti 
[(0, RELIGION, LIFE
  '0.006*"god" + 0.006*"man" + 0.005*"us" + 0.004*"upon" + 0.003*"shall" + '
  '0.003*"life" + 0.003*"men" + 0.003*"death" + 0.003*"lord" + 0.003*"must"'),
 (1, LOVE, TIME
  '0.018*"love" + 0.010*"time" + 0.010*"never" + 0.008*"baby" + 0.008*"way" + '
  '0.008*"want" + 0.007*"come" + 0.007*"life" + 0.007*"back" + 0.007*"away"'),
 (2, PEOPLE, WORLD
  '0.005*"people" + 0.004*"new" + 0.003*"state" + 0.003*"may" + 0.002*"time" + '
  '0.002*"world" + 0.002*"also" + 0.002*"first" + 0.002*"court" + '
  '0.002*"american"'),
 (3, HIPHOP/RAP
  '0.010*"shit" + 0.009*"fuck" + 0.009*"nigga" + 0.008*"man" + 0.007*"ya" + '
  '0.006*"bitch" + 0.006*"cause" + 0.006*"niggas" + 0.006*"money" + '
  '0.005*"back"')]"""

"""def create_model():
    songs = pd.read_csv('../Genius_song_lyrics_DB/song_lyrics.csv', nrows=20)

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
    # print(data_words)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    for text in texts:
        print(text)
        break
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
    # lda_model.save(fname="Genius_song_lyrics_DB/lda_model/lda_mod")

    # visualize_topics(lda_model, num_topics, corpus, id2word)
"""