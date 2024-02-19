from datasets import DatasetDict, Dataset
import pandas as pd
import numpy as np
import emb_util

def get_dataset(dataset_name):
    if(dataset_name in emb_util.similarity_datasets):
        path_train = f'data/sts/{dataset_name}_test.csv'
        path_test = f'data/sts/{dataset_name}_test.csv'
        # path_train = f'data/temp/{dataset_name}.csv'
        # path_test = f'data/temp/{dataset_name}.csv'
    elif(dataset_name in emb_util.unsplitted_datasets):
        path_train = f'data/cls/{dataset_name}.csv'
        path_test = f'data/cls/{dataset_name}.csv'
    elif(dataset_name in emb_util.splitted_datasets):
        path_train = f'data/cls/{dataset_name}_train.csv'
        path_test = f'data/cls/{dataset_name}_test.csv'

    df_train = pd.read_csv(path_train, sep='\t')
    df_test = pd.read_csv(path_test, sep='\t')

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "test": Dataset.from_pandas(df_test)
    })

    return dataset


def get_small_dataset(dataset_name, instance_number=100):
    '''
    arguments:
        instance_number: number of desired instances from original dataset for training data size.
    '''
    dataset = get_dataset(dataset_name)

    # Create a smaller training dataset for faster training times
    df_train = pd.DataFrame(dataset['train']).sample(frac=1).reset_index(drop=True)
    per_class_train = np.round(instance_number / df_train.label.nunique()).astype(int)
    df_train = df_train.groupby('label').head(per_class_train).sample(frac=1).reset_index(drop=True)

    # Create a smaller test dataset for faster training times
    df_test = pd.DataFrame(dataset['test']).sample(frac=1).reset_index(drop=True)
    per_class_test = np.round((instance_number*3) / df_test.label.nunique()).astype(int)
    df_test = df_test.groupby('label').head(per_class_test).sample(frac=1).reset_index(drop=True)

    small_dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "test": Dataset.from_pandas(df_test),
    })

    return small_dataset