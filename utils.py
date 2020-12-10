import pandas as pd
import emoji
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import preprocessor
from imblearn.under_sampling import RandomUnderSampler
from sklearn import feature_selection, metrics, model_selection, preprocessing

lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.TweetTokenizer()


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

def remove_urls(text):
    preprocessor.set_options('urls')
    return preprocessor.clean(text)


def remove_emojis(text):
    preprocessor.set_options('emojis', 'smiles')
    return preprocessor.clean(text)


def remove_hashtags(text):
    preprocessor.set_options('hashtags')
    return preprocessor.clean(text)


def remove_mentions(text):
    preprocessor.set_options('mentions')
    return preprocessor.clean(text)


def remove_standard(text):
    preprocessor.set_options('reserved_words', 'numbers', 'escape_chars')
    text = re.sub(r'\+(\d|\.)*', ' ', text)

    return preprocessor.clean(text)


def remove_punctuation(text):
    text = re.sub(r'([^\w\s])^(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])', ' ', text)

    return text


def lemmatize_text(text):
    return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((text))]


def preprocess(text, flg_lemm=True, lst_stopwords=None, flg_remove_mentions=None,
              flg_remove_emojis=None, flg_remove_hashtags=None, flg_remove_punctuation=None, flg_remove_urls=None):

    text = text.replace('<LH>', ' ')
    text = remove_standard(text)
    text = " ".join(emoji.get_emoji_regexp().split(text))
    text = text.lower()

    if flg_remove_mentions:
        text = remove_mentions(text)

    if flg_remove_emojis:
        text = remove_emojis(text)

    if flg_remove_hashtags:
        text = remove_hashtags(text)

    if flg_remove_urls:
        text = remove_urls(text)

    if flg_remove_punctuation:
        text = remove_punctuation(text)

    text = text.strip()

    if lst_stopwords is not None:
        lst_text = text.split()
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]
        text = " ".join(lst_text)

    if flg_lemm:
        text = lemmatize_text(text)
        text = " ".join(text)

    return text


def generate_tags(df, hashtag_threshold = 200):
    mlb = preprocessing.MultiLabelBinarizer(sparse_output=True)
    add_tag = np.vectorize(lambda x: f"hashtag_{x}")
    hashtags = pd.DataFrame(mlb.fit_transform(df.hashtags).toarray(), columns=add_tag(mlb.classes_))
    hashtag_freq = hashtags.sum(axis=0).sort_values(ascending=False)
    hashtags = hashtags.filter(hashtag_freq.iloc[:hashtag_threshold].index)

    return hashtags


def split_dataset(df):
    ## split dataset
    train, test = model_selection.train_test_split(df[df['identification'] == 'train'], test_size=0.1)

    print(f"Train: {train.shape}, test: {test.shape}")

    return train, test


def undersample(X, y):
    rus = RandomUnderSampler(random_state=0)
    return rus.fit_resample(X, y)


def select_features(X_train, y_train, vect, n_features, ngram_range, threshold):
    vectorizer = vect(max_features=n_features, ngram_range=ngram_range)

    vectorizer.fit(X_train)
    X_train_vect = vectorizer.transform(X_train)
    dic_vocabulary = vectorizer.vocabulary_

    print(f'Whole vocabolary {len(dic_vocabulary)}')

    if threshold > 0:
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

        for cat in np.unique(y_train):
            print("# {}:".format(cat))
            print("  . selected features:",
                  len(dtf_features[dtf_features["emotion"] == cat]))
            print("  . top features:", ",".join(
                dtf_features[dtf_features["emotion"] == cat]["feature"].values[:10]))
            print(" ")

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
    f1_score = metrics.f1_score(y_test, predicted, average='weighted')
    auc = metrics.roc_auc_score(y_test, predicted_prob,
                                multi_class="ovr")
    print("F1-score:", round(f1_score, 2))
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


def clean_text(df):
    # remove URL
    df['text'] = df['text'].str.replace(r'<LH>', r'')
    df['text'] = df['text'].str.replace(r'http(\S)+', r'')
    df['text'] = df['text'].str.replace(r'http ...', r'')
    df['text'] = df['text'].str.replace(r'http', r'')
    df[df['text'].str.contains(r'http')]
    # remove RT, @
    df['text'] = df['text'].str.replace(r'(RT|rt)[ ]*@[ ]*[\S]+', r'')
    df[df['text'].str.contains(r'RT[ ]?@')]
    df['text'] = df['text'].str.replace(r'@[\S]+', r'')
    # remove non-ascii words and character
    df['text'] = df['text'].str.replace(r'_[\S]?', r'')
    # remove &, < and >
    df['text'] = df['text'].str.replace(r'&amp;?', r'and')
    df['text'] = df['text'].str.replace(r'&lt;', r'<')
    df['text'] = df['text'].str.replace(r'&gt;', r'>')
    # remove extra space
    df['text'] = df['text'].str.replace(r'[ ]{2, }', r' ')
    # insert space between punctuation marks
    df['text'] = df['text'].str.replace(r'([\w\d]+)([^\w\d ]+)', r'\1 \2')
    df['text'] = df['text'].str.replace(r'([^\w\d ]+)([\w\d]+)', r'\1 \2')
    # insert space between emojis
    #     df['text'] = df['text'].str.replace(emoji.get_emoji_regexp(), r'\1 ')
    # lower case and strip white spaces at both ends
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.strip()

    return df