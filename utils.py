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
from sklearn import feature_selection, feature_extraction, metrics, model_selection, naive_bayes, pipeline, manifold, preprocessing
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


def load_data(n_rows=None):
    data_id = pd.read_csv('./data/data_identification.csv')
    emo = pd.read_csv('./data/emotion.csv')

    tweets_str = open('./data/tweets_DM.json').read()
    tweets_str = "[" + tweets_str.replace("}\n{", "}\n,{") + "]"
    tweets = json.loads(tweets_str)

    if n_rows is None:
        tweet_df = pd.DataFrame(tweets[:n_rows])
    else:
        tweet_df = pd.DataFrame(tweets[:n_rows])
    source = pd.DataFrame(list(tweet_df._source.apply(lambda x: x['tweet']).values))
    tweet_df = pd.concat([tweet_df, source], axis=1).drop(['_source', '_type', '_index'], axis=1)

    tweet_df = tweet_df.merge(data_id, left_on='tweet_id', right_on='tweet_id')
    tweet_df = tweet_df.merge(emo, how='left')

    print(f"{tweet_df.shape} records loaded")

    return tweet_df


# def clean_text(text):
#     # remove numbers
#     text_nonum = re.sub(r'\d+', '', text)
#     # remove punctuations and convert characters to lower case
#     text_nopunct = "".join([char.lower() for char in text_nonum if char not in string.punctuation])
#     # substitute multiple whitespace with single whitespace
#     # Also, removes leading and trailing whitespaces
#     text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
#     return text_no_doublespace


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'([^\w\s])^(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text


def preprocess(text, flg_stemm, flg_lemm, stopwords=None):
    text = text.str.replace('<LH>', '')
    text = text.apply(emoji.get_emoji_regexp().split).apply(" ".join)

    return text.apply(
        lambda x: utils_preprocess_text(
            x,
            flg_stemm=flg_stemm,
            flg_lemm=flg_lemm,
            lst_stopwords=stopwords
        )
    )


def split_dataset(df):
    ## split dataset
    train, test = model_selection.train_test_split(df[df['identification'] == 'train'], test_size=0.3)
    ## get target
    y_train = train["emotion"].values
    y_test = test["emotion"].values

    X_train = train['clean_text']
    X_test = test['clean_text']

    print(f"Train: {X_train.shape}, test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def select_features(X_train, y_train, vect, n_features, ngram_range, threshold):
    vectorizer = vect(max_features=n_features, ngram_range=ngram_range)

    vectorizer.fit(X_train)
    X_train_vect = vectorizer.transform(X_train)
    dic_vocabulary = vectorizer.vocabulary_

    print(f'Whole vocabolary {len(dic_vocabulary)}')

    X_names = vectorizer.get_feature_names()
    p_value_limit = threshold
    dtf_features = pd.DataFrame()
    for cat in np.unique(y_train):
        chi2, p = feature_selection.chi2(X_train_vect, y_train == cat)
        dtf_features = dtf_features.append(pd.DataFrame(
            {"feature": X_names, "score": 1 - p, "emotion": cat}))
        dtf_features = dtf_features.sort_values(["emotion", "score"],
                                                ascending=[True, False])
        dtf_features = dtf_features[dtf_features["score"] > p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()

    # for cat in np.unique(y_train):
    #     print("# {}:".format(cat))
    #     print("  . selected features:",
    #           len(dtf_features[dtf_features["emotion"] == cat]))
    #     print("  . top features:", ",".join(
    #         dtf_features[dtf_features["emotion"] == cat]["feature"].values[:10]))
    #     print(" ")

    vectorizer = vect(vocabulary=X_names)
    vectorizer.fit(X_train)
    X_train_vect = vectorizer.transform(X_train)
    dic_vocabulary = vectorizer.vocabulary_

    print(f'Reduced vocabolary {len(dic_vocabulary)}')

    return X_train_vect, dic_vocabulary, vectorizer


def evaluate(y_test, predicted, predicted_prob):
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob,
                                multi_class="ovr")
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i],
                                                 predicted_prob[:, i])
        ax[0].plot(fpr, tpr, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i],
                                                      metrics.auc(fpr, tpr))
                   )
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_array[:, i], predicted_prob[:, i])
        ax[1].plot(recall, precision, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i],
                                                      metrics.auc(recall, precision))
                   )
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall',
              ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()