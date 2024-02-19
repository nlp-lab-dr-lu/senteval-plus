from tqdm import tqdm
import os
import logging

import emb_util
from data import *

from angle_emb import AnglE, Prompts

class Angle_Embeddings:
    def __init__(self, model_name, datasets):

        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        
        self.model_name = model_name
        self.datasets = datasets

        for dataset in self.datasets:
            print('>>>>>>>>', self.model_name, dataset, '<<<<<<<<')
            self.dataset_name = dataset
            dataset = get_dataset(dataset)
            self.train_data, self.test_data = dataset["train"], dataset["test"]
            if(self.dataset_name in emb_util.unsplitted_datasets or self.dataset_name in emb_util.similarity_datasets):
                embeddings = self.get_embeddings(self.train_data, self.dataset_name, self.model_name, True)
            elif(self.dataset_name in emb_util.splitted_datasets):
                train_embeddings = self.get_embeddings(self.train_data, self.dataset_name+'_train', self.model_name, True)
                test_embeddings = self.get_embeddings(self.test_data, self.dataset_name+'_test', self.model_name, True)
            else:
                raise Exception("unknown dataset")
            
    def get_embeddings(self, dataset, split, angle_type, save):
        logging.info('encoding data and generating embeddings for test/train')
        '''
            dataset: a hugging face dataset object
            split: the dataset name string to save embeddings in a path
            save: T/F value to save the model in results directory
            type: angle-bert or angle-llama
        '''
        if(angle_type=='angle-bert'):
            angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg', device="cuda:1").cuda()
        elif(angle_type=='angle-llama'):
            angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf', pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2')
        
        angle.set_prompt(prompt=Prompts.A)
        
        embeddings = []
        for data_row in tqdm(dataset):
            data_row['text'] = '' if data_row['text'] == None else data_row['text']
            embedding = angle.encode([{'text': data_row['text']}], to_numpy=True)
            embeddings.append(embedding[0])
        
        if (save):
            emb_util.save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)

        return embeddings
