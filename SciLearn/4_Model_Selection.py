import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
n_samples = 3000
X_train = news.data[:n_samples]
y_train = news.target[:n_samples]

print X_train
print y_train

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def get_stop_word():
    result = set()
    for line in open('data/stop_words_en.txt', 'r').readline():
        result.add(line.strip())
    return result

stop_words = get_stop_word()
# print stop_words

clf = Pipeline(
    [
        ('vect', TfidfVectorizer(
            stop_words=stop_words,
            token_pattern=ur"\b[a-z0-9_\-\.]+[a-z0-9_\-\.]+\b",
        )),
        ('nb', MultinomialNB(alpha=0.01))
    ]
)

from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(len(X_train), K, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print scores
    print ("Mean score: {0:.3f} ( +/- {0:.3f})").format(np.mean(scores), sem(scores))

evaluate_cross_validation(clf, X_train, y_train, 3)

def calc_params(X, y, clf, param_values, param_name, K):
    train_scores = np.zeros(len(param_values))
    test_scores = np.zeros(len(param_values))
    for i, param_value in enumerate(param_values):
        print param_name, ' = ', param_value
        clf.set_params(**{param_name:param_value})

        k_train_scores = np.zeros(K)
        k_test_scores = np.zeros(K)

        cv = KFold(n_samples, K, shuffle=True, random_state=0)
        for j, (train, test) in enumerate(cv):
            clf.fit([X[k] for k in train], y[train])
            k_train_scores[j] = clf.scores([X[k] for k in train], y[train])
            k_test_scores[j] = clf.scores([X[k] for k in test], y[test])
        train_scores[i] = np.mean(k_train_scores)
        test_scores[i] = np.mean(k_test_scores)

