from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import emb_util
from emb_util import logger, save_embeddings
from data import get_dataset

class SBert_Embeddings:
    def __init__(self, model_name, datasets):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
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
            embedding = self.model.encode(data_row['text'])
            embeddings.append(np.array(embedding))
        if (save):
            save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)
            
        return embeddings