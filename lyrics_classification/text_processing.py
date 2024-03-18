import joblib
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import gensim
import pandas as pd
from Config import Constants
import spacy as spacy
from gensim.utils import simple_preprocess
from nltk.stem import PorterStemmer

WORD = re.compile(r'\w+')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = stopwords.words('english')
stop_words.extend(['1', '2', 'yeah', 'la', 'ah', 'oh', 'na', 'da',
                   'hey', 'got', 'go', 'one', 'em', 'v', 'let', 'ooh', 'would', 'say', 'take', 'see', 'said',
                   'chorus', 'intro', 'verse', 'produced'])
stopwords_dict = Counter(stop_words)


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(texts):
    # stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc))
             if word not in stopwords_dict] for doc in texts]


def regTokenize(text):
    words = WORD.findall(text)
    return words


def sort_tuples(tuples, key_idx):
    # Step 1: Define a key function that returns the desired element of each tuple.
    key_func = lambda x: x[key_idx]

    # Step 2: Use the `sorted` function to sort the list of tuples using the key function.
    sorted_tuples = sorted(tuples, key=key_func)

    return sorted_tuples


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


def lemmatization(spacy_nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = spacy_nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def print_pickle(file):
    topic_model = joblib.load("/WASABI_DB/topics/lda_model_16.jl")
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


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    #Separating sentences when word starting with uppercase letter, except I*, is found
    text = re.sub(r"([A-Z]+)","<stop>\\1", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(",", "<stop>")
    text = text.replace("and", "<stop>")
    text = text.replace(".", "<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("  ", "<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    sentences = [sentence for sentence in sentences if 3 <= len(re.findall(r'\w+', sentence)) <= 10]
    return sentences


def compute_text_similarity(file1, file2):
    nlp = spacy.load("en_core_web_lg")
    with open(file1) as f1:
        text1 = f1.read()
    with open(file2) as f2:
        text2 = f2.read()
    text1 = """
I found a love, for me
Darling, just dive right in and follow my lead
Well, I found a girl, beautiful and sweet
"""
    text2 ="""  It's late in the evening
She's wondering what clothes to wear
She puts on her makeup
And brushes her long blonde hair
And then she asks me
"""
    # text1 = preprocess_classifier_text(text1)
    #text2 = preprocess_classifier_text(text2)
    # nlp = en_core_web_sm.load()
    sentences_1 = split_into_sentences(text1)
    sentences_2 = split_into_sentences(text2)
    for sentence1 in sentences_1:
        for sentence2 in sentences_2:
            s1 = nlp(sentence1)
            s2 = nlp(sentence2)
            sim = s1.similarity(s2)
            if sim > 0.85:
                print(sim)
                print(sentence1)
                print(sentence2)
    """t1 = nlp(text1)
    t2 = nlp(text2)
    print(t1.similarity(t2))"""