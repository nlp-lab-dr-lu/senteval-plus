import os
import sys
import json
import numpy as np
import pandas as pd
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import emb_util

# from whitening import whiten
from .whitening import Whitens
from .validation import InnerKFoldMLPClassifier, InnerKFoldClassifier
from IsoScore import IsoScore

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

class WhiteningEvaluation:
    def __init__(self, config):
        """
            config: Dict
                EMBEDDINGS_PATH: path to load the embeddings
                RESULTS_PATH: path to save results
                drawcm: whether to draw the confusion matrix or not
                classifier: type of classifier for classification task ['lr', 'rf', 'nv', 'svm']
                whitenings_methods: type of whitening methods for classification task ['pca', 'zca', 'pca-cor', 'zca-cor', 'cholesky']
                kfold: number of folds for k fold inner classification
                encoders: list of encoders to evaluate
                datasets: list of datasets for evaluation ["mr", "cr", "subj", "mpqa", "trec", "mrpc", "sstf", "bbbp"]
        """
        self.EMBEDDINGS_PATH = 'embeddings/' if 'EMBEDDINGS_PATH' not in config else config['EMBEDDINGS_PATH']
        self.RESULTS_PATH = 'results/' if 'RESULTS_PATH' not in config else config['RESULTS_PATH']
        self.drawcm = False if 'drawcm' not in config else config['drawcm']
        self.classifier = 'mlp' if 'classifier' not in config else config['classifier']
        self.whitening_methods = ['pca', 'zca', 'pca_cor', 'zca_cor', 'cholesky'] if 'whitenings_methods' not in config else config['whitenings_methods']
        self.kfold = 5
        list_of_encoders = ["bert", "all-mpnet-base-v2", "simcse", "angle-bert", "angle-llama", "llama-7B", "llama2-7B", "text-embedding-3-small"]
        self.encoders = list_of_encoders if 'encoders' not in config else config['encoders']

        if 'datasets' not in config:
            raise Exception("No datasets decleard")
        else:
            self.datasets = config['datasets']

        self.results = []

    def run(self):
        for dataset in self.datasets:
            for encoder in self.encoders:
                X, y, nclasses = self.load_data(dataset, encoder)
                for method in self.whitening_methods:
                    print(f"\n<<evaluating {dataset.capitalize()} with {method.upper()} whitened {encoder.capitalize()} using {self.classifier.upper()} classifier>>")
                    trf = Whitens().fit(X, method = method)
                    X_whitened = trf.transform(X) # X is whitened
                    acc, acc_list, f1_w = self.sentEval(X_whitened, y, self.kfold, self.classifier, nclasses, dataset, method, encoder)
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
                        'IScore': IScore.item(),
                        'accuracy_list': acc_list,
                        'kfold': self.kfold
                    }
                    self.results.append(result)

                json_object = json.dumps(self.results, indent=4)
                with open(f"{self.RESULTS_PATH}/{dataset}_eval/{self.classifier}_eval_results3.json", "w") as outfile:
                    outfile.write(json_object)
        return self.results

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
            dev_accuracy, test_accuracy, test_f1_w, testresults_acc, cm_data = clf.run()
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

        return test_accuracy, testresults_acc, test_f1_w

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
        path = f'{self.EMBEDDINGS_PATH}{dataset_name}/{encoder_name}_{dataset_name}'
        if (dataset_name in emb_util.splitted_datasets):
            data_train = pd.read_csv(path+'_train_embeddings.csv' ,sep='\t')
            data_test = pd.read_csv(path+'_test_embeddings.csv' ,sep='\t')
            data = pd.concat([data_train, data_test], axis=0)
        elif(dataset_name in emb_util.unsplitted_datasets and dataset_name=='bbbp'):
            data = pd.read_csv('https://jlu.myweb.cs.uwindsor.ca/embeddings/MolNet/bbbp/llama2-7B_bbbp_embeddings.csv', index_col='Unnamed: 0')
        elif(dataset_name in emb_util.unsplitted_datasets):
            data = pd.read_csv(path+'_embeddings.csv', sep='\t')
        else:
            raise Exception("Could'nt find embeddings file")

        data = emb_util.check_and_reorder_dataframe(data)
        # get labels only
        labels = []
        for i, data_row in data.iterrows():
            labels.append(data_row['label'])
        y = np.array(labels)
        # getting embeddings only
        X = np.array(data.iloc[:, 2:])

        # get number of classes
        classes = np.unique(y)
        nclasses = len(classes)

        return X, y, nclasses