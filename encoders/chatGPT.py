from tqdm import tqdm
import numpy as np
import emb_util
from emb_util import logger, save_embeddings
from data import get_dataset
import openai

class ChatGPT_Embeddings:
    def __init__(self, model_name, datasets):
        self.model_name = "text-embedding-3-small"
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
            data_row['text'] = '' if data_row['text'] == None else data_row['text']
            response=openai.Embedding.create(
                model=self.model_name,
                input=data_row['text'])
            embedding = [item["embedding"] for item in response["data"]]
            embeddings.append(np.array(embedding[0]))

        if (save):
            save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)

        return embeddings
