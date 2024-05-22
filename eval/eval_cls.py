import os
import sys
import json
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import emb_util
from emb_util import logger

from .validation import InnerKFoldMLPClassifier, InnerKFoldClassifier
from IsoScore import IsoScore



class Evaluation:
    def __init__(self, config):
        """
            config: Dict
                EMBEDDINGS_PATH: path to load the embeddings
                RESULTS_PATH: path to save results
                drawcm: whether to draw the confusion matrix or not
                classifier: type of classifier for classification task ['lr', 'rf', 'nv', 'svm']
                kfold: number of folds for k fold inner classification
                encoders: list of encoders to evaluate
                datasets: list of datasets for evaluation ["mr", "cr", "subj", "mpqa", "trec", "mrpc", "sstf", "bbbp"]
        """
        
        self.EMBEDDINGS_PATH = 'embeddings/' if 'EMBEDDINGS_PATH' not in config else config['EMBEDDINGS_PATH']
        self.RESULTS_PATH = 'results/' if 'RESULTS_PATH' not in config else config['RESULTS_PATH']
        self.drawcm = False if 'drawcm' not in config else config['drawcm']
        self.classifier = 'mlp' if 'classifier' not in config else config['classifier']
        self.eval_whitening = True
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
                logger.info(f"evaluating {dataset.upper()} dataset with {encoder.upper()} using {self.classifier.upper()} classifier")
                X, y, nclasses = self.load_data(dataset, encoder)
                IScore = IsoScore.IsoScore(X)
                acc, acc_list, f1_w, roc_auc = self.sentEval(X, y, self.kfold, self.classifier, nclasses, dataset, 'No Whitening', encoder)
                logger.info(f'accuracy : {acc} , weighted f1 : {f1_w}')
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

                json_object = json.dumps(self.results, indent=4)
                path = f"{self.RESULTS_PATH}/{dataset}_eval/{self.classifier}_eval_results_temp.json"
                logger.info(f'saving evaluation results in "{path}"')
                with open(path, "w") as outfile:
                    outfile.write(json_object)
                logger.info(f'The evaluation successfully completed.')
        return self.results

    def sentEval(self, X, y, kfold, classifier, nclasses, dataset, whitening_method, encoder):
        cls_config = {
            'nclasses': nclasses,
            'seed': random.randint(1, 100),
            'classifier': classifier,
            'kfold': kfold,
            'drawcm': False
        }
        if(classifier == 'mlp'):
            clf = InnerKFoldMLPClassifier(X, y, cls_config)
            if(self.drawcm):
                self.draw_cm(cm_data, dataset, whitening_method, encoder)
        elif(classifier in ['lr', 'rf', 'svm', 'nb']):
            clf = InnerKFoldClassifier(X, y, cls_config)
        else:
            raise Exception("unknown classifier")
        
        dev_accuracy, test_accuracy, test_f1_w, test_rocauc, testresults_acc, cm_data = clf.run()

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
        logger.info('svaing cm plot in ',path)
        fig.savefig(path, format='pdf')

    def load_data(self, dataset_name, encoder_name):
        path = f'{self.EMBEDDINGS_PATH}/{dataset_name}/{encoder_name}_{dataset_name}'
        if(dataset_name == 'mrpc'):
            base_path = f'{self.EMBEDDINGS_PATH}/{dataset_name}/{encoder_name}_{dataset_name}1'
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

            X1 = np.array(data1.iloc[:, 1:])
            X2 = np.array(data2.iloc[:, 1:])
            X = np.c_[np.abs(X1 - X2), X1 * X2]
            X = pd.DataFrame(X)
            y = pd.DataFrame(y)
            y = y.rename(columns={0: 'label'})
            X.insert(0, 'label', y['label'])
            X.insert(0, 'text', 'text')
            data = X
        elif(dataset_name in emb_util.splitted_datasets):
            data_train = pd.read_csv(path+'_train_embeddings.csv' ,sep='\t')
            data_test = pd.read_csv(path+'_test_embeddings.csv' ,sep='\t')
            data = pd.concat([data_train, data_test], axis=0)
        elif(dataset_name in emb_util.unsplitted_datasets):
            data = pd.read_csv(path+'_embeddings.csv', sep='\t')
        else:
            raise Exception(f"Could'nt find embeddings file in {path}")
        data = emb_util.check_and_reorder_dataframe(data)
        labels = []
        for i, data_row in data.iterrows():
            labels.append(data_row['label'])
        y = np.array(labels)
        X = np.array(data.iloc[:, 2:])
        classes = np.unique(y)
        nclasses = len(classes)

        return X, y, nclasses