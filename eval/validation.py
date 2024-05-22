from __future__ import absolute_import, division, unicode_literals

import logging
import numpy as np
import torch
from .classifier import MLP

import sklearn
assert (sklearn.__version__ >= "0.18.0"), \
    "need to update sklearn to version >= 0.18.0"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score #For the binary case will return F1 score for pos_label
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
        self.testaucs = []
        self.usepytorch = True if 'usepytorch' not in config else config['usepytorch']
        self.classifier_config = { 'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 5 }
        self.k = 5 if 'kfold' not in config else config['kfold']
        self.device = 'cuda'

    def run(self):
        regs = [10 ** t for t in range(-5, -1)]
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
            # logging.info('Best param found at split {0}: l2reg = {1} with score {2}'.format(count, optreg, np.max(scores)))
            self.devresults.append(np.max(scores))

            clf = MLP(self.classifier_config,
                      inputdim=self.featdim,
                      nclasses=self.nclasses,
                      l2reg=optreg,
                      seed=self.seed)
            clf.fit(X_train, y_train, validation_split=0.05)
            # calculate accuracy
            self.testaccs.append(round(100 * clf.score(X_test, y_test), 2))
            # calculate f1
            y_pred = clf.predict(X_test)
            self.testf1s.append(round(100 * f1_score(y_test, y_pred, average='weighted')))
            cm_data = {'y_test': y_test, 'y_pred': y_pred}

        devaccuracy = round(np.mean(self.devresults), 2)
        testaccuracy = round(np.mean(self.testaccs), 2)
        testf1 = round(np.mean(self.testf1s), 2)

        return devaccuracy, testaccuracy, testf1, 'N/A', self.testf1s, cm_data

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
        self.testaucs = []
        self.modelname = config['classifier']
        self.k = 5 if 'kfold' not in config else config['kfold']

    def run(self):
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        count = 0
        for train_idx, test_idx in skf.split(self.X, self.y):
            count += 1
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            scores = []

            if self.modelname == 'lr':
                regs = [2 ** t for t in range(-2, 4, 1)]
                scores = []
                # for reg in regs:
                #     regscores = []
                #     for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                #         X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                #         y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                #         clf = LogisticRegression(C=reg, random_state=self.seed, max_iter=5000)
                #         clf.fit(X_in_train, y_in_train)
                #         regscores.append(clf.score(X_in_test, y_in_test))
                #     scores.append(round(100 * np.mean(regscores), 2))
                optreg = regs[np.argmax(scores)] if len(scores) > 0 else 1.0
                clf = LogisticRegression(C=optreg, random_state=self.seed, max_iter=5000)
            elif self.modelname == 'svm':
                clf = make_pipeline(StandardScaler(), SVC(random_state=self.seed, gamma='scale'))
            elif self.modelname == 'rf':
                clf = RandomForestClassifier(random_state=self.seed)
            elif self.modelname == 'nb':
                clf = GaussianNB()
            else:
                raise Exception("unknown classifier")

            clf.fit(X_train, y_train)
            # calculate accuracy
            self.testaccs.append(round(100 * clf.score(X_test, y_test), 2))
            # calculate f1
            y_pred = clf.predict(X_test)
            self.testf1s.append(round(100 * f1_score(y_test, y_pred, average='weighted')))
            cm_data = {'y_test': y_test, 'y_pred': y_pred}
            # calculate roc
            # y_pred_proba = clf.predict_proba(X_test)[:,1]
            # auc = roc_auc_score(y_test, y_pred_proba, average='weighted')
            auc = 0
            self.testaucs.append(round(100 * auc))
            # print(f'\tAUC of {count} fold: {auc}')

        # devaccuracy = round(np.mean(self.devresults), 2)
        devaccuracy = None
        testaccuracy = round(np.mean(self.testaccs), 2)
        testf1 = round(np.mean(self.testf1s), 2)
        testauc = round(np.mean(self.testaucs), 2)

        return devaccuracy, testaccuracy, testf1, testauc, self.testf1s, cm_data