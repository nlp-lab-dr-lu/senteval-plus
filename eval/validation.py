from __future__ import absolute_import, division, unicode_literals

import logging
import numpy as np
from .classifier import MLP
from .hyperparameter import HyperParameters

import sklearn
assert (sklearn.__version__ >= "0.18.0"), \
    "need to update sklearn to version >= 0.18.0"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score #For the binary case will return F1 score for pos_label
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_classif_name(classifier_config, usepytorch):
    if not usepytorch:
        modelname = 'sklearn-LogReg'
    else:
        nhid = classifier_config['nhid']
        optim = 'adam' if 'optim' not in classifier_config else classifier_config['optim']
        bs = 64 if 'batch_size' not in classifier_config else classifier_config['batch_size']
        modelname = 'pytorch-MLP-nhid%s-%s-bs%s' % (nhid, optim, bs)
    return modelname


# Pytorch version
class InnerKFoldMLPClassifier(object):

    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.featdim = X.shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.devresults = []
        self.testaccs = []
        self.testf1s = []
        self.usepytorch = config['usepytorch']
        self.classifier_config = config['classifier']
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.k = 5 if 'kfold' not in config else config['kfold']

    def run(self):
        logging.info('Training {0} with (inner) {1}-fold cross-validation'.format(self.modelname, self.k))

        regs = [10 ** t for t in range(-5, -1)] if self.usepytorch else \
            [2 ** t for t in range(-2, 4, 1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True,
                                   random_state=self.seed)
        count = 0
        for train_idx, test_idx in skf.split(self.X, self.y):
            count += 1
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            scores = []
            for reg in regs:
                regscores = []
                for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                    X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                    y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                    clf = MLP(self.classifier_config,
                              inputdim=self.featdim,
                              nclasses=self.nclasses,
                              l2reg=reg,
                              seed=self.seed)
                    clf.fit(X_in_train, y_in_train,validation_data=(X_in_test, y_in_test))

                    regscores.append(clf.score(X_in_test, y_in_test))
                scores.append(round(100 * np.mean(regscores), 2))
            optreg = regs[np.argmax(scores)]
            logging.info('Best param found at split {0}: l2reg = {1} with score {2}'.format(count, optreg, np.max(scores)))
            self.devresults.append(np.max(scores))

            clf = MLP(self.classifier_config,
                      inputdim=self.featdim,
                      nclasses=self.nclasses,
                      l2reg=optreg,
                      seed=self.seed)
            clf.fit(X_train, y_train, validation_split=0.05)
            y_pred = clf.predict(X_test)
            self.testaccs.append(round(100 * clf.score(X_test, y_test), 2))
            self.testf1s.append(round(100 * f1_score(y_test, y_pred, average='weighted')))
            cm_data = {'y_test': y_test, 'y_pred': y_pred}

        devaccuracy = round(np.mean(self.devresults), 2)
        testaccuracy = round(np.mean(self.testaccs), 2)
        testf1 = round(np.mean(self.testf1s), 2)

        return devaccuracy, testaccuracy, testf1, self.testf1s, cm_data

class InnerKFoldClassifier(object):

    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.featdim = X.shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.devresults = []
        self.testaccs = []
        self.testf1s = []
        self.modelname = config['classifier']
        self.k = 5 if 'kfold' not in config else config['kfold']

    def run(self):
        logging.info('Training {0} with (inner) {1}-fold cross-validation'.format(self.modelname, self.k))

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        count = 0
        for train_idx, test_idx in skf.split(self.X, self.y):
            count += 1
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            scores = []

            if self.modelname == 'lr':
                # hp = HyperParameters('lr')
                # conf = hp.get_hyperparameters()
                # logistic regression will have hypertuning
                regs = [2 ** t for t in range(-2, 4, 1)]
                scores = []
                for reg in regs:
                    regscores = []
                    for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                        X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                        y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                        clf = LogisticRegression(C=reg, random_state=1111, max_iter=1000000)
                        clf.fit(X_in_train, y_in_train)
                        regscores.append(clf.score(X_in_test, y_in_test))
                    scores.append(round(100 * np.mean(regscores), 2))
                optreg = regs[np.argmax(scores)]
                logging.info('Best param found at split {0}: l2reg = {1} with score {2}'.format(count, optreg, np.max(scores)))
                self.devresults.append(np.max(scores))
                clf = LogisticRegression(C=optreg, random_state=self.seed, max_iter=1000000)
            elif self.modelname == 'svm':
                # scaler = StandardScaler()
                # X_train = scaler.fit_transform(X_train)
                # X_test = scaler.transform(X_test)
                # clf = SVC(kernel='linear', C=1, cache_size=5000, random_state=self.seed)
                clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1, class_weight='balanced'))
            elif self.modelname == 'rf':
                deps = [3, 5, 7, 10]
                scores = []
                for dep in deps:
                    depscores = []
                    for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                        X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                        y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                        clf = RandomForestClassifier(max_depth=dep, random_state=1111)
                        clf.fit(X_in_train, y_in_train)
                        depscores.append(clf.score(X_in_test, y_in_test))
                    scores.append(round(100 * np.mean(depscores), 2))
                optdep = deps[np.argmax(scores)]
                logging.info('Best depth found at split {0}: depth = {1} with score {2}'.format(count, optdep, np.max(scores)))
                self.devresults.append(np.max(scores))
                clf = RandomForestClassifier(max_depth=optdep, random_state=self.seed)
            elif self.modelname == 'nb':
                clf = GaussianNB()
            else:
                raise Exception("unknown classifier")

            clf.fit(X_train, y_train)
            self.testaccs.append(round(100 * clf.score(X_test, y_test), 2))
            y_pred = clf.predict(X_test)
            self.testf1s.append(round(100 * f1_score(y_test, y_pred, average='weighted')))
            cm_data = {'y_test': y_test, 'y_pred': y_pred}

        # devaccuracy = round(np.mean(self.devresults), 2)
        devaccuracy = None
        testaccuracy = round(np.mean(self.testaccs), 2)
        testf1 = round(np.mean(self.testf1s), 2)

        return devaccuracy, testaccuracy, testf1, self.testf1s, cm_data