import logging
import json
from tqdm import tqdm
import emb_util

import torch
from transformers import BertModel, AutoTokenizer

import numpy as np
import pandas as pd

from data import *


class Bert_Embeddings:
    def __init__(self, model_name, datasets):

        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

        logging.info('loading model and tokenizer')
        # The base Bert Model transformer outputting raw hidden-states without any specific head on top.
        self.model_name = model_name
        self.model = BertModel.from_pretrained("bert-base-uncased",
                                                # output_hidden_states = True
                                                ) # Whether the model returns all hidden-states.)
        self.model.to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        logging.info('loading datasets')
        self.datasets = datasets
        for dataset in self.datasets:
            print('>>>>>>>>', self.model_name, dataset, '<<<<<<<<')
            self.dataset_name = dataset
            dataset = get_dataset(dataset) # small data: dataset = get_small_dataset("mr", 10)
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
            save_embeddings: T/F value to save the model in results directory
        '''
        embeddings = []
        for data_row in tqdm(dataset):
            data_row['text'] = '' if data_row['text'] == None else data_row['text']
            tokens = self.tokenizer.encode_plus(
                data_row['text'],
                add_special_tokens=True,
                max_length=64, # to use maximum length: max_length=self.tokenizer.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokens['input_ids'].to("cuda")
            token_type_ids = tokens['token_type_ids'].to("cuda")
            attention_mask = tokens['attention_mask'].to("cuda")

            # Obtain sentence embedding
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask = attention_mask
                )
                embedding = torch.mean(outputs.last_hidden_state, dim=1)

                embedding = embedding.squeeze().to("cpu")
                embeddings.append(np.array(embedding))

        if (save):
            emb_util.save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)
        
        return embeddings

# if __name__ == "__main__":
#     bert_embeddings = Bert_Embeddings("bert", "imdb")

# investigating layers:
# hidden_states = outputs[2]
# print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
# layer_i = 0
# print ("Number of batches:", len(hidden_states[layer_i]))
# batch_i = 0
# print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
# token_i = 0
# print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

# print(data_row['text'])
# print(outputs.last_hidden_state.shape)
# print(embedding.shape)
# embedding = torch.mean(outputs[2][1:], dim=1)