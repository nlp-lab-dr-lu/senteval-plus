import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn
from sklearn.decomposition import PCA
import sklearn.cluster
import emb_util
from emb_util import logger
from .whitening import Whitens
from sklearn.metrics import silhouette_score, davies_bouldin_score

class ClusteringEvaluation:
    def __init__(self, config):
        self.EMBEDDINGS_PATH = 'embeddings/' if 'EMBEDDINGS_PATH' not in config else config['EMBEDDINGS_PATH']
        self.RESULTS_PATH = 'results/' if 'RESULTS_PATH' not in config else config['RESULTS_PATH']
        self.eval_whitening = True
        list_of_encoders = ["bert", "all-mpnet-base-v2", "simcse", "angle-bert", "angle-llama", "llama-7B", "llama2-7B", "text-embedding-3-small"]
        self.encoders = list_of_encoders if 'encoders' not in config else config['encoders']
        self.whitening_methods = ['pca', 'zca', 'pca_cor', 'zca_cor', 'cholesky'] if 'whitenings_methods' not in config else config['whitenings_methods']
        self.clustering = 'mini_batch_k_means'
        if 'datasets' not in config:
            raise Exception("No datasets decleard")
        else:
            self.datasets = config['datasets']
        self.results = []
        self.json_results = []

    def run(self):
        for dataset in self.datasets:
            for encoder in self.encoders:
                logger.info(f"evaluating {dataset.upper()} dataset with {encoder.upper()} using MiniBatchKMeans clustering")
                X, y, nclasses = self.load_data(dataset, encoder)
                v_measure_list = []
                for i in range(0, 5):
                    clustering_model = sklearn.cluster.MiniBatchKMeans(
                        n_clusters=nclasses,
                        batch_size=500,
                        n_init="auto",
                    )
                    clustering_model.fit(X)
                    v_measure = round(silhouette_score(X, clustering_model.labels_ ) * 100, 2)
                    # v_measure = round(sklearn.metrics.cluster.v_measure_score(y, clustering_model.labels_ ) * 100, 2)
                    v_measure_list.append(v_measure)
                    # self.results.append(((encoder, v_measure, X, clustering_model.labels_)))
                json_result = {
                    'dataset': dataset,
                    'encoder': encoder,
                    'clustering': self.clustering,
                    'whitening': '',
                    'v_measure': v_measure_list,
                }
                logger.info(f"V-measure: {v_measure_list}")
                self.json_results.append(json_result)

                for method in self.whitening_methods:
                    logger.info(f"evaluating {dataset.upper()} dataset with {method.upper()} whitened {encoder.upper()} using MiniBatchKMeans clustering")
                    trf = Whitens().fit(X, method = method)
                    X_whitened = trf.transform(X) # X is whitened
                    v_measure_list = []
                    for i in range(0, 5):
                        clustering_model = sklearn.cluster.MiniBatchKMeans(
                            n_clusters=nclasses,
                            batch_size=5000,
                            n_init="auto",
                        )
                        clustering_model.fit(X_whitened)
                        v_measure = round(silhouette_score(X_whitened, clustering_model.labels_ ) * 100, 2)
                        # v_measure = round(sklearn.metrics.cluster.v_measure_score(y, clustering_model.labels_ ) * 100, 2)
                        v_measure_list.append(v_measure)
                        # self.results.append(((encoder, v_measure, X, clustering_model.labels_)))
                    json_result = {
                        'dataset': dataset,
                        'encoder': encoder,
                        'clustering': self.clustering,
                        'whitening': method,
                        'v_measure': v_measure_list,
                    }
                    logger.info(f"V-measure: {v_measure_list}")
                    self.json_results.append(json_result)

                json_object = json.dumps(self.json_results, indent=4)
                path = f"{self.RESULTS_PATH}/{dataset}_eval/{self.clustering}_eval_results_temp.json"
                logger.info(f'saving evaluation results in "{path}"')
                with open(path, "w") as outfile:
                    outfile.write(json_object)

        # self.draw_plot()

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
        print(nclasses)

        return X, y, nclasses


    def draw_plot(self):
        encoder_names = {
            'text-embedding-3-small': 'ChatGPT',
            'bert': 'BERT',
            'all-mpnet-base-v2': 'SBERT',
            'simcse': 'SimCSE',
            'llama-7B': 'LLaMA',
            'llama2-7B': 'LLaMA2',
            'angle-llama': 'AnglE-LLaMA',
            'angle-bert': 'AnglE-BERT',
        }
        self.adjust_fonts()
        plot_rows = len(self.encoders)
        plot_cols = int(len(self.results)/len(self.encoders))

        colors = ['midnightblue', 'royalblue','midnightblue','royalblue','midnightblue','powderblue','deepskyblue']
        cmap = ListedColormap(colors)

        fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(9, 14))
        fig.subplots_adjust(hspace=0.25)

        for z, (encoder, v_measure, X, clusters) in enumerate(self.results):
            i, j = divmod(z, plot_cols)
            # print(z, i, j)
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
            if plot_rows == 1 or plot_cols == 1:
                ax = axs[j] if plot_rows == 1 else axs[i]
            else:
                ax = axs[i, j]
            ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap=cmap, marker='o', s=2.5)
            if(j==0):
                axs[i, j].set_ylabel(encoder_names[encoder].upper(), fontsize=10)
                whitening = 'no whitening'
            else:
                parts = encoder.split('-')
                whitening = parts[-1]
            axs[i,j].set_xlabel(v_measure)
            if(i==0):
                ax.set_title(whitening.upper(), fontsize=10)
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig('cluserting.png', format='png', dpi=500)

    def adjust_fonts(self):
        # Fix fonts
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