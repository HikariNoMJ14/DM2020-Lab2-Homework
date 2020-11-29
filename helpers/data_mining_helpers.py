import nltk
import pandas as pd
import string

"""
Helper functions for data mining lab session 2018 Fall Semester
Author: Elvis Saravia
Email: ellfae@gmail.com
"""


def format_rows(docs):
    """ format the text field and strip special characters """
    D = []
    for d in docs.data:
        temp_d = " ".join(d.split("\n")).strip('\n\t')
        D.append([temp_d])
    return D


def format_labels(target, docs):
    """ format the labels """
    return docs.target_names[target]


def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter += 1
    return ("The amoung of missing records is: ", counter)


def tokenize_text(text, remove_stopwords=False, remove_punctuation=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []

    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):

            if remove_stopwords and word in nltk.corpus.stopwords.words('english'):
                continue

            if remove_punctuation and word in string.punctuation:
                continue

            tokens.append(word)

    return tokens


def get_frequencies(data, score, source):
    docs_to_words = data[data['special_score'] == score]
    docs_to_words = docs_to_words[docs_to_words['special_source'] == source]
    word_frequencies = pd.DataFrame(docs_to_words.drop(['special_score', 'special_source'], axis=1).sum(axis=0),
                                    columns=['frequency'])

    return word_frequencies[word_frequencies > 0].dropna()
