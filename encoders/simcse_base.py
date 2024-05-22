from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import emb_util
from emb_util import logger, save_embeddings
from data import get_dataset


class SimCSE_Embeddings:
    def __init__(self, model_name, datasets):
        # The base Bert Model transformer outputting raw hidden-states without any specific head on top.
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        self.datasets = datasets

        for dataset in self.datasets:
            logger.info(f'encoding {dataset} dataset with {self.model_name} model')
            self.dataset_name = dataset
            dataset = get_dataset(dataset) 
            self.train_data, self.test_data = dataset["train"], dataset["test"]
            if(self.dataset_name in emb_util.splitted_datasets or self.dataset_name in emb_util.bio_datasets):
                train_embeddings = self.get_embeddings(self.train_data, self.dataset_name+'_train', True)
                test_embeddings = self.get_embeddings(self.test_data, self.dataset_name+'_test', True)
            else:
                embeddings = self.get_embeddings(self.train_data, self.dataset_name, True)
                
    def get_embeddings(self, dataset, split, save):
        '''
            dataset: a hugging face dataset object
            split: the split name string to save embeddings in a path
            save: T/F value to save the model in results directory
        '''

        embeddings = []
        for i, data_row in tqdm(dataset.iterrows()):
        
            tokens = self.tokenizer(data_row['text'], padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                embedding = self.model(**tokens, output_hidden_states=True, return_dict=True).pooler_output

            embeddings.append(np.array(embedding[0]))
        
        if (save):
            save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)

        return embeddings