# from datasets import DatasetDict, Dataset
import pandas as pd
import numpy as np
import emb_util

def get_dataset(dataset_name):
    if(dataset_name in emb_util.similarity_datasets):
        path_train = f'data/sts_datasets/{dataset_name}_test.csv'
        path_test = f'data/sts_datasets/{dataset_name}_test.csv'
    elif(dataset_name in emb_util.unsplitted_datasets):
        path_train = f'data/tcls_datasets/{dataset_name}.csv'
        path_test = f'data/tcls_datasets/{dataset_name}.csv'
    elif(dataset_name in emb_util.splitted_datasets):
        path_train = f'data/tcls_datasets/{dataset_name}_train.csv'
        path_test = f'data/tcls_datasets/{dataset_name}_test.csv'
    elif(dataset_name in emb_util.bio_datasets):
        path_train = f'data/bio_data/{dataset_name}/{dataset_name}.csv'
        path_test = f'data/bio_data/{dataset_name}/{dataset_name}.csv'
    elif(dataset_name in emb_util.clustering_datasets):
        path_train = f'data/tclu_datasets/{dataset_name}.csv'
        path_test = f'data/tclu_datasets/{dataset_name}.csv'
    else:
        raise Exception("unknown dataset")


    df_train = pd.read_csv(path_train, sep='\t')
    df_test = pd.read_csv(path_test, sep='\t')

    dataset = {
        "train": df_train,
        "test": df_test
    }

    return dataset