import numpy as np
import pandas as pd
from itertools import combinations
import logging

splitted_datasets = ["yelpp", "imdb", "agnews", "yelpf", "trec", "sstf", "mrpc", "mrpc1", "mrpc2"]
unsplitted_datasets = ["mr", "cr", "subj", "mpqa"]
similarity_datasets = ["sts1", "sts2"]

def save_embeddings(embeddings, dataset, model_name, dataset_name, split):
    logging.info('saving data embeddings')
    embeddings = np.array(embeddings)
    data = pd.concat([pd.DataFrame(dataset), pd.DataFrame(embeddings)], axis=1)
    if(dataset_name in ['sts1', 'sts2', 'mrpc1', 'mrpc2']):
        dataset_name = dataset_name[:-1]

    path = f'embeddings/{dataset_name}/{model_name}_{split}_embeddings.csv'
    # path = f'embeddings/sts/{self.model_name}_{split}_embeddings.csv'
    data.to_csv(path, sep='\t', index=False)

def get_layers_mean(model_name, dataset, embeddings_list):
    # numbers = list(range(1, 33))
    # all_tuples = list(combinations(numbers, 2))
    # datasets = ["sts1", "sts2"]
    # model_names = ["llama2-7B"]
    # for model_name in model_names:
    #     for dataset in datasets:
    #         for tpl in all_tuples:
    #             get_layers_mean(model_name, dataset, [tpl[0], tpl[1]])
    base_path = 'results/embeddings_sts/llama_layers'
    path1 = f'{base_path}/{model_name}-layer-{embeddings_list[0]}_{dataset}_test_embeddings.csv'
    path2 = f'{base_path}/{model_name}-layer-{embeddings_list[1]}_{dataset}_test_embeddings.csv'

    df1 = pd.read_csv(path1, sep='\t')
    df2 = pd.read_csv(path2, sep='\t')
    sentences = df1['text'] # save sentences for later
    df1 = df1.drop('text', axis=1)
    df2 = df2.drop('text', axis=1)

    average_df = pd.DataFrame((df1.values + df2.values) / 2)
    average_df = pd.concat([sentences, average_df], axis=1)
    average_df.to_csv(f'results/embeddings_sts/llama_pair/{model_name}-pair-{embeddings_list[0]}-and-{embeddings_list[1]}_{dataset}_test_embeddings.csv', index=False, sep='\t')
    print(f'saved average of {embeddings_list[0]} and {embeddings_list[1]} for {dataset}')
    return





