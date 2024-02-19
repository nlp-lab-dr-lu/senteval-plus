import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

import emb_util
from data import *

import openai

class ChatGPT_Embeddings:
    def __init__(self, model_name, datasets):
        openai.api_key  = ('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')
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

        # The base Bert Model transformer outputting raw hidden-states without any specific head on top.
        self.model_name = "text-embedding-3-small"
        self.datasets = datasets

        for dataset in self.datasets:
            print('>>>>>>>>', self.model_name, dataset, '<<<<<<<<')
            self.dataset_name = dataset
            dataset = get_dataset(dataset)
            self.train_data, self.test_data = dataset["train"], dataset["test"]
            if(self.dataset_name in emb_util.unsplitted_datasets or self.dataset_name in emb_util.similarity_datasets):
                embeddings = self.get_embeddings(self.train_data, self.dataset_name, True)
            elif(self.dataset_name in emb_util.splitted_datasets):
                train_embeddings = self.get_embeddings(self.train_data, self.dataset_name+'_train', True)
                test_embeddings = self.get_embeddings(self.test_data, self.dataset_name+'_test', True)
            else:
                raise Exception("unknown dataset")
            
    def get_embeddings(self, dataset, split, save):
        logging.info('encoding data and generating embeddings for test/train')
        '''
            dataset: a hugging face dataset object
            split: the split name string to save embeddings in a path
            save: T/F value to save the model in results directory
        '''

        embeddings = []
        for data_row in tqdm(dataset):
            data_row['text'] = '' if data_row['text'] == None else data_row['text']
            # Sentences are encoded by calling model.encode()
            response=openai.Embedding.create(
                model=self.model_name,
                input=data_row['text'])
            embedding = [item["embedding"] for item in response["data"]]
            embeddings.append(np.array(embedding[0]))

        if (save):
            emb_util.save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)

        return embeddings
