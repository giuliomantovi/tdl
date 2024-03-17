from text_processing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

def preprocess_classifier_text(text):
    tokens = regTokenize(text)
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopwords_dict]
    # stemmed removed for classifier creations because it takes too much time
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)


def create_text_classifier():
    # Genre inference training the classifier with genius lyrics and tag (genre)
    cont = 0
    x_preprocessed = []
    y_preprocessed = []
    y = any
    for chunk in pd.read_csv('../Genius_song_lyrics_DB/song_lyrics.csv',
                             engine='c', chunksize=100000, usecols=['lyrics', 'tag']):
        if cont == 20:
            break
        print(chunk)
        cont += 1
        x, y = chunk.lyrics, chunk.tag
        y_preprocessed += [genre for genre in y]
        x_preprocessed += [preprocess_classifier_text(text) for text in x]
    # print(x_preprocessed)

    vectorizer = TfidfVectorizer()
    x_transformed = vectorizer.fit_transform(x_preprocessed)
    classifier = MultinomialNB()
    classifier.fit(x_transformed, y_preprocessed)

    # save the model to disk (current one trained on first 2 million songs, 0.46% accuracy)
    classifier_filename = '../Genius_song_lyrics_DB/genre_classifier.sav'
    pickle.dump(classifier, open(classifier_filename, 'wb'))

    vectorizer_filename = '../Genius_song_lyrics_DB/vectorizer.pk'
    pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
    # evaluate_text_classifier()


def evaluate_text_classifier():
    # testing classifier accuracy with the 21th chunk of genius lyrics (which has not been used to train the classifier)
    loaded_classifier = pickle.load(open("../Genius_song_lyrics_DB/genre_classifier.sav", 'rb'))
    loaded_vectorizer = pickle.load(open("../Genius_song_lyrics_DB/vectorizer.pk", 'rb'))

    cont = 0
    x_preprocessed = []
    y_preprocessed = []
    y = any

    for chunk in pd.read_csv('../Genius_song_lyrics_DB/song_lyrics.csv', engine='c', chunksize=100000):
        if cont == 20:
            # print(chunk)
            x, y = chunk.lyrics, chunk.tag
            y_preprocessed += [genre for genre in y]
            x_preprocessed += [preprocess_classifier_text(text) for text in x]
            break
        cont += 1

    x_test_transformed = loaded_vectorizer.transform(x_preprocessed)
    y_pred = loaded_classifier.predict(x_test_transformed)
    accuracy = accuracy_score(y_preprocessed, y_pred)
    print("Accuracy:", accuracy)


def predict_genre_from_lyrics():
    # load the model from disk
    loaded_classifier = pickle.load(open("../Genius_song_lyrics_DB/genre_classifier.sav", 'rb'))
    loaded_vectorizer = pickle.load(open("../Genius_song_lyrics_DB/vectorizer.pk", 'rb'))

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
    # print(X_test_transformed)
    result = loaded_classifier.predict(X_test_transformed)
    print(result)

    """summaries = pd.read_pickle("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/topics/song_id_to_topics.pickle")
    print(summaries["ObjectId(5714deed25ac0d8aee57e542)"])
    obj = pd.read_pickle("C:/Users/Utente/UNI/tesina_LAUREA/WASABI_DB/id_to_summary_lines.pickle")
    print(obj["5714deed25ac0d8aee57e542"])"""
    """[(4, 0.05279845), (8, 0.06588726), (13, 0.38976097), (16, 0.022661319), (18, 0.022483176), (27, 0.010201428), (30, 0.054924987), (32, 0.016545713), (40, 0.010497759), (41, 0.04243385), (44, 0.020272728), (47, 0.010643412), (48, 0.013438855), (51, 0.025851179), (52, 0.013056019), (55, 0.0149408085), (56, 0.010882135), (58, 0.017950308)]
["it ain't nothing wrong but y'all know me", "i'm a loc and my loc niggaz need me bitch", "when i'm in the club i don't see real niggaz", "i see a bunch of wannabe's in it"]"""

