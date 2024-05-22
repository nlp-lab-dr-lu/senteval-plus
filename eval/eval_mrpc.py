import os
import sys
import json
import numpy as np
import pandas as pd
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from whitening import whiten
from .whitening import Whitens
from .validation import InnerKFoldMLPClassifier, InnerKFoldClassifier
from IsoScore import IsoScore

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, config):
        self.EMBEDDINGS_PATH = 'embeddings/' if 'EMBEDDINGS_PATH' not in config else config['EMBEDDINGS_PATH']
        self.RESULTS_PATH = 'results/' if 'RESULTS_PATH' not in config else config['RESULTS_PATH']
        self.drawcm = False if 'drawcm' not in config else config['drawcm']
        self.classifier = 'mlp' if 'classifier' not in config else config['classifier']
        self.whitening_methods = ['pca', 'zca', 'pca_cor', 'zca_cor', 'cholesky'] if 'whitenings_methods' not in config else config['whitenings_methods']
        self.kfold = 5
        list_of_encoders = ["bert", "all-mpnet-base-v2", "simcse", "angle-bert", "angle-llama", "llama-7B", "llama2-7B", "text-embedding-3-small"]
        self.encoders = list_of_encoders if 'encoders' not in config else config['encoders']
        self.datasets = ['mrpc']
        self.results = []
        
    def run(self):
        for dataset in self.datasets:
            for encoder in self.encoders:
                print(f"\n<<evaluating {dataset.upper()} with {encoder.upper()} using {self.classifier.upper()} classifier>>")
                X1, X2, y, nclasses = self.load_data(dataset, encoder)
                X = np.c_[np.abs(X1 - X2), X1 * X2]
                IScore = IsoScore.IsoScore(X)

                acc, acc_list, f1_w, roc_auc = self.sentEval(X, y, self.kfold, self.classifier, nclasses, dataset, 'No Whitening', encoder)
                print(f'\twhitening method:  : {acc} : {f1_w}')
                result = {
                    'dataset': dataset,
                    'encoder': encoder,
                    'classfier': self.classifier,
                    'whitening': '',
                    'accuracy': acc,
                    'f1_weighted': f1_w,
                    'rocauc_weighted': roc_auc,
                    'IScore': IScore.item(),
                    'accuracy_list': acc_list,
                    'kfold': self.kfold
                }
                self.results.append(result)

                for method in self.whitening_methods:
                    print(f"\n<<evaluating {dataset.upper()} with {method.upper()} whitened {encoder.upper()} using {self.classifier.upper()} classifier>>")
                    trf = Whitens().fit(X, method = method)
                    X_whitened = trf.transform(X) # X is whitened
                    acc, acc_list, f1_w, roc_auc = self.sentEval(X_whitened, y, self.kfold, self.classifier, nclasses, dataset, method, encoder)
                    print(f'\twhitening method: {method} : {acc} : {f1_w}')
                    IScore = IsoScore.IsoScore(X_whitened)

                    method = 'zca-cor' if method == 'zca_cor' else method # make sure to be consistant in file names
                    method = 'pca-cor' if method == 'pca_cor' else method # make sure to be consistant in file names
                    result = {
                        'dataset': dataset,
                        'encoder': encoder,
                        'classfier': self.classifier,
                        'whitening': method,
                        'accuracy': acc,
                        'f1_weighted': f1_w,
                        'rocauc_weighted': roc_auc,
                        'IScore': IScore.item(),
                        'accuracy_list': acc_list,
                        'kfold': self.kfold
                    }
                    self.results.append(result)

                json_object = json.dumps(self.results, indent=4)
                with open(f"{self.RESULTS_PATH}/{dataset}_eval/{self.classifier}_eval_results1.json", "w") as outfile:
                    outfile.write(json_object)

    def sentEval(self, X, y, kfold, classifier, nclasses, dataset, whitening_method, encoder):
        if(classifier == 'mlp'):
            classifier = {
                'nhid': 0,
                'optim': 'rmsprop',
                'batch_size': 128,
                'tenacity': 3,
                'epoch_size': 5
            }
            config = {
                'nclasses': nclasses,
                'seed': random.randint(1, 100),
                'usepytorch': True,
                'classifier': classifier,
                'nhid': classifier['nhid'],
                'kfold': kfold,
                'drawcm': False
            }
            clf = InnerKFoldMLPClassifier(X, y, config)
            dev_accuracy, test_accuracy, test_f1_w, test_rocauc, testresults_acc, cm_data = clf.run()
            if(self.drawcm):
                self.draw_cm(cm_data, dataset, whitening_method, encoder)
        elif(classifier in ['lr', 'rf', 'svm', 'nb']):
            config = {
                'nclasses': nclasses,
                'seed': random.randint(1, 100),
                'classifier': self.classifier,
                'kfold': self.kfold,
                'drawcm': False
            }
            clf = InnerKFoldClassifier(X, y, config)
            dev_accuracy, test_accuracy, test_f1_w, testresults_acc, cm_data = clf.run()
        else:
            raise Exception("unknown classifier")

        return test_accuracy, testresults_acc, test_f1_w, test_rocauc

    def draw_cm(self, cm_data, dataset, whitening_method, encoder):
        cm = confusion_matrix(cm_data['y_test'], cm_data['y_pred'], labels=np.unique(cm_data['y_test']))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(cm_data['y_test']))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_axis_off()
        ax.set_title(whitening_method.upper())
        plt.rcParams.update({'font.size': 30})
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
        path = f'{self.RESULTS_PATH}/{dataset}_eval/cm/{encoder}-{whitening_method}-cm.pdf'
        print('svaing cm plot',path)
        fig.savefig(path, format='pdf')

    def load_data(self, dataset_name, encoder_name):
        base_path = f'{self.EMBEDDINGS_PATH}{dataset_name}/{encoder_name}_{dataset_name}1'
        data_train_1 = pd.read_csv(base_path+'_train_embeddings.csv' ,sep='\t')
        data_test_1 = pd.read_csv(base_path+'_test_embeddings.csv' ,sep='\t')
        data1 = pd.concat([data_train_1, data_test_1], axis=0)

        base_path = f'{self.EMBEDDINGS_PATH}{dataset_name}/{encoder_name}_{dataset_name}2'
        data_train_2 = pd.read_csv(base_path+'_train_embeddings.csv' ,sep='\t')
        data_test_2 = pd.read_csv(base_path+'_test_embeddings.csv' ,sep='\t')
        data2 = pd.concat([data_train_2, data_test_2], axis=0)

        labels_train = pd.read_csv('./data/tcls_datasets/mrpcs_train.csv' ,sep='\t')
        labels_test = pd.read_csv('./data/tcls_datasets/mrpcs_test.csv' ,sep='\t')
        y = pd.concat([labels_train, labels_test], axis=0)
        y = np.array(y).squeeze()

        # getting embeddings only
        X1 = np.array(data1.iloc[:, 1:])
        X2 = np.array(data2.iloc[:, 1:])

        # get number of classes
        classes = np.unique(y)
        nclasses = len(classes)

        return X1, X2, y, nclasses