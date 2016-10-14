import numpy as np
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
boston = load_boston()

print (boston.data.shape)
print (type(boston.data))
print (boston.feature_names)
print (np.max(boston.target), np.min(boston.target), np.mean(boston.target))
print (boston.DESCR)
print "\n\n First row:"
print boston.data[1]
print "\n\n Target:"
print boston.target[1]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=0)

from sklearn.feature_selection import *
fs = SelectKBest(score_func=f_regression, k=5)
X_new=fs.fit_transform(X_train, y_train)
print (zip(fs.get_support(), boston.feature_names))

x_min, x_max = X_new[:, 0].min() - .5, X_new[:, 0].max() + .5
y_min, y_max = y_train.min() - .5, y_train.max() + .5

fig, axes = plt.subplots(1, 5)
fig.set_size_inches(24, 24)

for i in range(5):
    axes[i].set_aspect('equal')
    axes[i].set_title('Feature ' + str(i))
    axes[i].set_xlabel('Feature')
    axes[i].set_ylabel('Median house value')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    plt.scatter(X_new[:, i], y_train)

# plt.show()


from sklearn.preprocessing import StandardScaler

scalerX = StandardScaler().fit(X_train)
print 'y_train:'
print y_train
X_train = scalerX.transform(X_train)
scalery = StandardScaler().fit(y_train.reshape(len(y_train), 1))
y_train = scalery.transform(y_train.reshape(len(y_train), 1))
X_test = scalerX.transform(X_test)
y_test = scalery.transform(y_test.reshape(len(y_test), 1))
# print (np.max(X_train), np.min(X_train), np.mean(X_train),
#        np.max(y_train), np.min(y_train), np.mean(y_train))


from sklearn.cross_validation import cross_val_score, KFold
def train_and_evaluate(clf, X_train, y_train):
    clf.fit(X_train, y_train)

    print ("Coefficient of determination on training set:", clf.score(X_train, y_train))

    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print ("Average coefficient of determination using 5-fold crossvalidation:", np.mean(scores))


from sklearn import linear_model
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=42)
train_and_evaluate(clf_sgd, X_train, y_train)
print "SGDRegressor:"
print (clf_sgd.coef_)


clf_sgd1 = linear_model.SGDRegressor(loss='squared_loss', penalty='l1', random_state=42)
train_and_evaluate(clf_sgd1, X_train, y_train)
print "SGDRegressor_l1:"
print (clf_sgd1.coef_)

clf_sgd2 = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', random_state=42)
train_and_evaluate(clf_sgd2, X_train, y_train)
print "SGDRegressorL2:"
print (clf_sgd2.coef_)


from sklearn import svm
clf_svr = svm.SVC(kernel='linear')
train_and_evaluate(clf_svr, X_train, y_train)
