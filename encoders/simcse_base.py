import logging
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import emb_util

from transformers import AutoModel, AutoTokenizer
from data import *


class SimCSE_Embeddings:
    def __init__(self, model_name, datasets):

        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

        logging.info('loading model and tokenizer')
        # The base Bert Model transformer outputting raw hidden-states without any specific head on top.
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
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
            tokens = self.tokenizer(data_row['text'], padding=True, truncation=True, return_tensors="pt")

            # Get the embeddings
            with torch.no_grad():
                embedding = self.model(**tokens, output_hidden_states=True, return_dict=True).pooler_output

            embeddings.append(np.array(embedding[0]))
        
        if (save):
            emb_util.save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)

        return embeddings