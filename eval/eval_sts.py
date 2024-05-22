import os
import json
import numpy as np
import pandas as pd
from .whitening import Whitens
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
from emb_util import logger

class Evaluation:
    def __init__(self, config):
        """
            config: Dict
                EMBEDDINGS_PATH: path to load the embeddings
                SCORES_PATH: path to load similarity scores
                RESULTS_PATH: path to save results
                encoders: list of encoders to evaluate
                datasets: list of datasets for evaluation ["stsb", "sts12", "sts13", "sts14", "sts15", "sts16"]
        """
        self.eval_whitening = True
        self.EMBEDDINGS_PATH = 'embeddings/' if 'EMBEDDINGS_PATH' not in config else config['EMBEDDINGS_PATH']
        self.SCORES_PATH = 'data/sts_datasets/'
        self.RESULTS_PATH = 'results/' if 'RESULTS_PATH' not in config else config['RESULTS_PATH']
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
                logger.info(f'calculating spearman correlation for {encoder} on {dataset} dataset')
                path_s1 = f'{self.EMBEDDINGS_PATH}/{dataset}/{encoder}_{dataset}1_embeddings.csv'
                path_s2 = f'{self.EMBEDDINGS_PATH}/{dataset}/{encoder}_{dataset}2_embeddings.csv'
                df_sent1 = pd.read_csv(path_s1, sep='\t')
                df_sent2 = pd.read_csv(path_s2, sep='\t')
                path_scores = f'{self.SCORES_PATH}/{dataset}s_test.csv'
                df_sts_scores = pd.read_csv(path_scores, sep='\t')
                df_scores = self.calculate_scores(df_sent1, df_sent2, df_sts_scores, encoder, dataset)

                p_val = self.pearson_value(df_scores) * 100
                s_val = self.spearman_value(df_scores) * 100

                result = {
                    "encoder": encoder,
                    "whitening": '',
                    "pearson_value": p_val,
                    "spearman_value": s_val
                }
                logger.info(f'whitening method: none, spearman value: {result["spearman_value"]}')
                self.results.append(result)

                if self.eval_whitening:
                    whitenings = ['zca', 'zca_cor', 'pca', 'pca_cor', 'cholesky']
                    for whitening in whitenings:
                        df_sent1_whitened, df_sent2_whitened = self.do_whitening(df_sent1, df_sent2, whitening)
                        whitening = 'zca-cor' if whitening == 'zca_cor' else whitening # make sure to be consistant in file names
                        whitening = 'pca-cor' if whitening == 'pca_cor' else whitening # make sure to be consistant in file names
                        encoder_name = 'wh' + whitening + '-' + encoder
                        df_scores = self.calculate_scores(df_sent1_whitened, df_sent2_whitened, df_sts_scores, encoder_name, dataset)

                        p_val = self.pearson_value(df_scores) * 100
                        s_val = self.spearman_value(df_scores) * 100

                        result = {
                            "encoder": encoder,
                            "whitening": whitening,
                            "pearson_value": p_val,
                            "spearman_value": s_val
                        }
                        logger.info(f'whitening method: {whitening}, spearman value: {result["spearman_value"]}')
                        self.results.append(result)

            json_object = json.dumps(self.results, indent=4)
            
            dir_path = f'{self.RESULTS_PATH}/{dataset}_eval/'
            if not os.path.exists(dir_path):
                # Create the directory
                os.makedirs(dir_path)
                logger.info(f"directory '{dir_path}' was created.")

            with open(dir_path+'eval_results1.json', 'w') as outfile:
                outfile.write(json_object)

    def get_cosine_similarity(self, feature_vec_1, feature_vec_2):
        output = cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))
        return output[0]

    def calculate_scores(self, df_sent1, df_sent2, df_scores, encoder, dataset):
        cos_sims = []
        for i, val in df_sent1.iterrows():
            sent1 = df_sent1.iloc[[i]]
            sent1 = sent1.iloc[:, 1:].to_numpy().squeeze()
            sent2 = df_sent2.iloc[[i]]
            sent2 = sent2.iloc[:, 1:].to_numpy().squeeze()
            cos_sim = self.get_cosine_similarity(sent1, sent2)
            cos_sim = np.float32(cos_sim)
            cos_sims.append(cos_sim[0])

        df_results = pd.DataFrame(cos_sims, columns=['cos_sim'])
        df_results = pd.concat([df_results, df_scores], axis=1)

        path = f'{self.RESULTS_PATH}/{dataset}_eval/cosim'
        if not os.path.exists(path):
            # Create the directory
            os.makedirs(path)
            logger.info(f"directory '{path}' was created.")
        df_results.to_csv(path+f'/{encoder}_{dataset}_test_cossim.csv', sep='\t', index=False)
        return df_results

    def spearman_value(self, df_scores):
        scor, p_val = spearmanr(df_scores["cos_sim"], df_scores["score"])
        return scor

    def pearson_value(self, df_scores):
        pcor, p_val = pearsonr(df_scores["cos_sim"], df_scores["score"])
        return pcor

    def calulate_anisotropy(self, df_scores):
        mean_cosine = np.mean(df_scores['cos_sim'])
        return mean_cosine

    def do_whitening(self, df_1, df_2, method):
        embeddings_1 = df_1.iloc[: ,1:]
        embeddings_2 = df_2.iloc[: ,1:]
        X = pd.concat([embeddings_1, embeddings_2])
        assert embeddings_1.shape[0] == embeddings_2.shape[0], "embedding size of two sentences are not the same"
        trf = Whitens().fit(X, method = method)
        X_whitened = trf.transform(X)
        data_size = df_1.shape[0]
        X_whitened_1 = X_whitened[:data_size]
        X_whitened_1 = pd.concat([df_1.iloc[: ,:1], pd.DataFrame(X_whitened_1)], axis=1)
        X_whitened_2 = X_whitened[data_size:]
        X_whitened_2 = pd.concat([df_2.iloc[: ,:1], pd.DataFrame(X_whitened_2)], axis=1)
        return X_whitened_1, X_whitened_2