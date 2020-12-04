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
from sklearn import feature_selection, feature_extraction, metrics, model_selection, ensemble, naive_bayes, pipeline, manifold, preprocessing
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


from utils import load_data, split_dataset, select_features, preprocess, evaluate


if __name__ == "__main__":
    n_rows = 250000
    flg_lemm = False
    flg_stemm = True
    n_features = 25000
    ngram_range = (1, 2)
    threshold = 0.99
    stopwords = nltk.corpus.stopwords.words("english")
    classifier = naive_bayes.MultinomialNB()
    # classifier = ensemble.RandomForestClassifier()
    vect = feature_extraction.text.TfidfVectorizer

    df = load_data(n_rows)
    df['clean_text'] = preprocess(
        df['text'],
        flg_lemm=flg_lemm,
        flg_stemm=flg_stemm,
        stopwords=stopwords
    )
    X_train, X_test, y_train, y_test = split_dataset(df)

    X_train_vect, vocab, vectorizer = select_features(
        X_train=X_train,
        y_train=y_train,
        vect=vect,
        n_features=n_features,
        ngram_range=ngram_range,
        threshold=threshold
    )

    features = pd.DataFrame(
        index=X_train.index,
        data=X_train_vect.toarray(),
        columns=vectorizer.get_feature_names()
    )

    print(features.head())

    ## pipeline
    model = pipeline.Pipeline([("vectorizer", vectorizer),
                               ("classifier", classifier)])
    ## train classifier
    model["classifier"].fit(X_train_vect, y_train)
    ## test
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)

    evaluate(y_test, predicted, predicted_prob)
