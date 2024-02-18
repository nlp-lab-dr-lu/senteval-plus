import os
import json
import numpy as np
import pandas as pd
import emb_util

# from whitening import whiten
from whitening import Whitens
from validation import InnerKFoldClassifier
from IsoScore import IsoScore

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

class Evlaution:
    def __init__(self):
        self.BASE_PATH = 'embeddings/'
        self.drawcm = False
        eval_mlp = True
        classifier = 'mlp' if eval_mlp else 'lr'
        eval_whitening = True
        results = []
        datasets = ["subj"] #"mr", "cr", "subj", "mpqa", "trec", "mrpc", "sstf"
        kfold = 5
        encoders = [
            "bert",
            "all-mpnet-base-v2",
            "simcse",
            "angle-bert", "angle-llama",
            "llama-7B", "llama2-7B",
            "text-embedding-3-small"
        ]

        for dataset in datasets:
            for encoder in encoders:
                print(f"<<evaluating {dataset} with {encoder} with {classifier}>>")
                X, y, nclasses = self.load_data(dataset, encoder)

                IScore = IsoScore.IsoScore(X)

                acc, acc_list = self.sentEval(X, y, kfold, classifier, nclasses, dataset, 'No Whitening', encoder)
                print(f'\twhitening method:  : {acc}')
                result = {
                    'dataset': dataset,
                    'encoder': encoder,
                    'classfier': classifier,
                    'whitening': '',
                    'accuracy': acc,
                    'IScore': IScore.item(),
                    'accuracy_list': acc_list,
                    'kfold': kfold
                }
                results.append(result)

                if eval_whitening:
                    whitenings_methods = ['pca', 'zca', 'zca_cor', 'pca_cor', 'cholesky'] 
                    for method in whitenings_methods:
                        trf = Whitens().fit(X, method = method)
                        X_whitened = trf.transform(X) # X is whitened
                        acc, acc_list = self.sentEval(X_whitened, y, kfold, classifier, nclasses, dataset, method, encoder)
                        print(f'\twhitening method: {method} : {acc}')
                        IScore = IsoScore.IsoScore(X_whitened)

                        method = 'zca-cor' if method == 'zca_cor' else method # make sure to be consistant in file names
                        method = 'pca-cor' if method == 'pca_cor' else method # make sure to be consistant in file names
                        result = {
                            'dataset': dataset,
                            'encoder': encoder,
                            'classfier': classifier,
                            'whitening': method,
                            'accuracy': acc,
                            'IScore': IScore.item(),
                            'accuracy_list': acc_list,
                            'kfold': kfold
                        }
                        results.append(result)

                json_object = json.dumps(results, indent=4)
                with open(f"results/{dataset}_eval/{classifier}_eval_results2.json", "w") as outfile:
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
                'seed': 2,
                'usepytorch': True,
                'classifier': classifier,
                'nhid': classifier['nhid'],
                'kfold': kfold,
                'drawcm': True
            }
            clf = InnerKFoldClassifier(X, y, config)
            dev_accuracy, test_accuracy, testresults_acc, cm_data = clf.run()
            if(self.drawcm):
                self.draw_cm(cm_data, dataset, whitening_method, encoder)


        elif(classifier == 'lr'):
            testresults_acc = []
            regs = [2**t for t in range(-2, 4, 1)]
            skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1111)
            innerskf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1111)

            for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                scores = []
                for reg in regs:
                    regscores = []
                    for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                        X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                        y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                        clf = LogisticRegression(C=reg, random_state=0, max_iter=100000)
                        clf.fit(X_in_train, y_in_train)
                        score = clf.score(X_in_test, y_in_test)
                        regscores.append(score)
                    # print(f'\t L2={reg} , fold {i} of {kfold}, score {score}')
                    scores.append(round(100*np.mean(regscores), 5))

                optreg = regs[np.argmax(scores)]
                # print('Best param found at split {0}:  L2 regularization = {1} with score {2}'.format(i, optreg, np.max(scores)))
                clf = LogisticRegression(C=optreg, random_state=0, max_iter=100000)
                clf.fit(X_train, y_train)

                f_acc = round(100*clf.score(X_test, y_test), 2)
                print(f'\taccuracy of {i} fold: {f_acc}')
                testresults_acc.append(f_acc)
            test_accuracy = round(np.mean(testresults_acc), 2)
        else:
            raise Exception("unknown classifier")

        return test_accuracy, testresults_acc

    def draw_cm(self, cm_data, dataset, whitening_method, encoder, download_font=False):
        if download_font:
            # add the font to Matplotlib
            # create the directory for fonts if it doesn't exist
            font_dir = "font/"
            os.makedirs(font_dir, exist_ok=True)

            # Download the font
            font_url = "http://foroogh.myweb.cs.uwindsor.ca/Times_New_Roman.ttf"
            font_path = os.path.join(font_dir, "times_new_roman.ttf")
            if not os.path.exists(font_path):
                os.system(f"wget -P {font_dir} {font_url}")
                
        font_files = font_manager.findSystemFonts(fontpaths=['font/'])
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)

        # Verify the font is recognized by Matplotlib
        font_name = "Times New Roman"
        if font_name in font_manager.get_font_names():
            print(f"'{font_name}' font successfully added.")
            # Set default font to Times New Roman
            matplotlib.rc('font', family=font_name)
        else:
            print(f"'{font_name}' font not found. Please check the font installation.")

        cm = confusion_matrix(cm_data['y_test'], cm_data['y_pred'], labels=np.unique(cm_data['y_test']))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(cm_data['y_test']))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_axis_off()
        ax.set_title(whitening_method.upper())
        plt.rcParams.update({'font.size': 30})
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
        path = f'results/{dataset}_eval/cm/{encoder}-{whitening_method}-cm.pdf'
        print('svaing cm plot',path)
        fig.savefig(path, format='pdf')

    def compute_scores(self, Y, y):
        # Calculate F1 and accuracy score
        f1 = f1_score(Y, y, average='micro')
        accuracy = accuracy_score(Y, y)
        # Calculate error rate
        cm = confusion_matrix(Y, y, labels=np.unique(Y))
        # print(cm)
        total_misclassified = sum(cm[i][j] for i in range(
            len(cm)) for j in range(len(cm)) if i != j)
        total_instances = sum(sum(row) for row in cm)
        # print(total_misclassified,total_instances)
        er = total_misclassified / total_instances

        return f1, accuracy, er

    def load_data(self, dataset_name, encoder_name):
        if (dataset_name in emb_util.splitted_datasets):
            base_path = f'{self.BASE_PATH}{dataset_name}/{encoder_name}_{dataset_name}'
            data_train = pd.read_csv(base_path+'_train_embeddings.csv' ,sep='\t')
            data_test = pd.read_csv(base_path+'_test_embeddings.csv' ,sep='\t')
            data = pd.concat([data_train, data_test], axis=0)
        elif(dataset_name in emb_util.unsplitted_datasets):
            path = f'{self.BASE_PATH}{dataset_name}/{encoder_name}_{dataset_name}_embeddings.csv'
            data = pd.read_csv(path, sep='\t')
        else:
            raise Exception("unknown dataset")

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

if __name__ == "__main__":
    eval = Evlaution()