import pandas as pd
import json
import functools
import emoji
import functools
import operator
import re
import nltk
import string

## for data
import json
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for processing
import re
import nltk
## for bag-of-words
from sklearn import feature_selection, neural_network, feature_extraction, metrics, model_selection, ensemble, naive_bayes, pipeline, manifold, preprocessing
## for explainer
from lime import lime_text
## for word embedding
import gensim
import gensim.downloader as gensim_api
## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
## for bert language model
import transformers


from utils import load_data, split_dataset, select_features, preprocess, evaluate, generate_tags


if __name__ == "__main__":
    n_rows = 250000
    flg_lemm = False
    flg_stemm = True
    flg_remove_punctuation = True
    flg_remove_urls = True
    flg_remove_emojis = True
    flg_remove_hashtags = True
    flg_remove_mentions = True
    n_features = 25000
    ngram_range = (1, 2)
    threshold = 0.95
    hashtag_threshold = 200
    stopwords = nltk.corpus.stopwords.words("english")
    model = naive_bayes.MultinomialNB(alpha=1)
    # model = ensemble.RandomForestClassifier()
    # model = neural_network.MLPClassifier()
    vect = feature_extraction.text.TfidfVectorizer

    df = load_data(n_rows)
    df['clean_text'] = df['text'].apply([lambda x: preprocess(
        x,
        flg_lemm=flg_lemm,
        flg_stemm=flg_stemm,
        flg_remove_punctuation=flg_remove_punctuation,
        flg_remove_urls=flg_remove_urls,
        flg_remove_emojis=flg_remove_emojis,
        flg_remove_hashtags=flg_remove_hashtags,
        flg_remove_mentions=flg_remove_mentions,
        lst_stopwords=stopwords
    )])

    X_train, X_test, y_train, y_test = split_dataset(df)

    X_train_vect, vocab, vectorizer = select_features(
        X_train=X_train,
        y_train=y_train,
        vect=vect,
        n_features=n_features,
        ngram_range=ngram_range,
        threshold=threshold
    )

    X_test_vect = vectorizer.transform(X_test)

    # train_hashtags = generate_tags(df[df['identification'] == 'train'])
    # test_hashtags = generate_tags(df[df['identification'] == 'test'])
    #
    # X_train_vect = pd.concat([X_train_vect, train_hashtags], axis=1)
    # X_test_vect = pd.concat([X_test_vect, test_hashtags], axis=1)

    print(vectorizer.get_feature_names()[:100])

    ## train classifier
    model.fit(X_train_vect, y_train)

    y_train_pred = model.predict(X_train_vect.toarray())
    acc_train = metrics.accuracy_score(y_true=y_train, y_pred=y_train_pred)

    print(f"Training accuracy: {acc_train}")

    ## test
    predicted = model.predict(X_test_vect)
    predicted_prob = model.predict_proba(X_test_vect)

    evaluate(y_test, predicted, predicted_prob)
