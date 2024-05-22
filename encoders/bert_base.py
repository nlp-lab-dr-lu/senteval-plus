from tqdm import tqdm
import torch
from transformers import BertModel, AutoTokenizer
import numpy as np
import emb_util
from emb_util import logger, save_embeddings
from data import get_dataset


class Bert_Embeddings:
    def __init__(self, model_name, datasets):

        self.model_name = model_name
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
            save_embeddings: T/F value to save the model in results directory
        '''
        embeddings = []
        for i, data_row in tqdm(dataset.iterrows()):
            data_row['text'] = ' ' if data_row['text'] == '' else data_row['text']
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
            save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)
        
        return embeddings