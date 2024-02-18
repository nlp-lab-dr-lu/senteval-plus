from tqdm import tqdm

import json
import numpy as np
import pandas as pd
# from whitening import whiten
from whitening import Whitens

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import spearmanr, pearsonr

import seaborn as sns
import matplotlib.pyplot as plt

class Evlaution:
    def __init__(self):
        eval_whitening = True
        BASE_PATH = 'embeddings/sts'
        self.BASE_PATH = BASE_PATH
        results = []

        encoders = [
            # "bert",
            # "all-mpnet-base-v2",
            # "simcse",
            # "angle-bert",
            "angle-llama",
            "llama-7B", "llama2-7B",
            "ChatGPT",
        ]
        tail = '_test_embeddings.csv'

        for encoder in encoders:
            print(f'calculating correlation for {encoder}')
            path_s1 = f'{BASE_PATH}/{encoder}_sts1{tail}'
            path_s2 = f'{BASE_PATH}/{encoder}_sts2{tail}'

            df_sent1 = pd.read_csv(path_s1, sep='\t')
            df_sent2 = pd.read_csv(path_s2, sep='\t')

            path_scores = f'{BASE_PATH}/stss_test.csv'
            df_sts_scores = pd.read_csv(path_scores, sep='\t')

            df_scores = self.calculate_scores(df_sent1, df_sent2, df_sts_scores, encoder)

            p_val = self.pearson_value(df_scores) * 100
            s_val = self.spearman_value(df_scores) * 100

            result = {
                "encoder": encoder,
                "whitening": '',
                "pearson_value": p_val,
                "spearman_value": s_val
            }
            results.append(result)

            if eval_whitening:
                whitenings = ['zca', 'zca_cor', 'pca', 'pca_cor', 'cholesky']
                for whitening in whitenings:
                    # if ((encoder == 'angle-llama' or encoder == 'llama-7B' or encoder == 'llama2-7B') and whitening == 'cholesky'):
                    #     continue
                    print(f'\twhitening method: {whitening}')
                    df_sent1_whitened, df_sent2_whitened = self.do_whitening(df_sent1, df_sent2, whitening)
                    whitening = 'zca-cor' if whitening == 'zca_cor' else whitening # make sure to be consistant in file names
                    whitening = 'pca-cor' if whitening == 'pca_cor' else whitening # make sure to be consistant in file names
                    encoder_name = 'wh' + whitening + '-' + encoder
                    df_scores = self.calculate_scores(df_sent1_whitened, df_sent2_whitened, df_sts_scores, encoder_name)

                    p_val = self.pearson_value(df_scores) * 100
                    s_val = self.spearman_value(df_scores) * 100

                    result = {
                        "encoder": encoder,
                        "whitening": whitening,
                        "pearson_value": p_val,
                        "spearman_value": s_val
                    }
                    results.append(result)

        print(results)
        json_object = json.dumps(results, indent=4)

        with open(f'results/sts_eval/eval_results.json', 'w') as outfile:
            outfile.write(json_object)



    def get_cosine_similarity(self, feature_vec_1, feature_vec_2):
        output = cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))
        return output[0]

    def calculate_scores(self, df_sent1, df_sent2, df_scores, encoder, save_scores=True):
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
        print('min found in cos_sim:', df_results['cos_sim'].min())
        print('min found in score:', df_results['score'].min())

        if save_scores:
            path = f'results/sts_eval/cosim/{encoder}_sts_test_cossim.csv'
            df_results.to_csv(path, sep='\t', index=False)
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
        # print('wh dim:', X_whitened.shape)
        X_whitened_1 = X_whitened[:1379]
        X_whitened_1 = pd.concat([df_1.iloc[: ,:1], pd.DataFrame(X_whitened_1)], axis=1)
        X_whitened_2 = X_whitened[1379:]
        X_whitened_2 = pd.concat([df_2.iloc[: ,:1], pd.DataFrame(X_whitened_2)], axis=1)
        return X_whitened_1, X_whitened_2

if __name__ == "__main__":
    eval = Evlaution()