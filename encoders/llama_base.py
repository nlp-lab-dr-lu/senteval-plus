from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, AutoTokenizer
from data import get_dataset
import emb_util
from emb_util import logger, save_embeddings
# from accelerate import infer_auto_device_map

class Llama_Embeddings:
    def __init__(self, model_name, datasets, strategy='layer', pooling=32):
        '''
            strategy: layer, pair, range
            pooling: could be an integer if strategy is range or layer, could be a tuple if strategy is pair
        '''
        self.pooling = pooling
        self.strategy = strategy

        logger.info('loading model and tokenizer')
        self.model_name = model_name
        models_path = {
            "llama-7B": "./llama_converted/7B",
            "llama2-7B": "./llama2_converted/7B",
            "llama3-8B": "./llama3_converted",
        }
        PATH_TO_CONVERTED_WEIGHTS = models_path[model_name]

        # Set device to auto to utilize GPU
        device = "auto"  # balanced_low_0, auto, balanced, sequential

        if self.model_name == "llama-310B":
            print("loading llama 30B takes much longer time due to GPU management issues.")
            self.model = LlamaForCausalLM.from_pretrained(
                PATH_TO_CONVERTED_WEIGHTS,
                device_map=device,
                max_memory={0: "12GiB", 1: "12GiB", 2: "12GiB", 3: "12GiB"},
                offload_folder="offload"
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                PATH_TO_CONVERTED_WEIGHTS,
                device_map=device,
                # output_hidden_states=True
            )
        # self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"
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
            data_row['text'] = ' ' if data_row['text'] == None else data_row['text']
            tokens = self.tokenizer(
                data_row['text'],
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=64,
                return_attention_mask = True
            )
            with torch.no_grad():
                output = self.model(**tokens, return_dict=True, output_hidden_states=True)
                hidden_states = output.hidden_states

                if self.strategy == 'layer' and isinstance(self.pooling, int):
                    embedding = (hidden_states[self.pooling]).mean(dim=1)
                elif self.strategy == 'range' and isinstance(self.pooling, int):
                    embedding = np.array(hidden_states[-self.pooling:]).mean(axis=0)
                    embedding = np.array(embedding).mean(axis=1)
                elif self.strategy == 'pair' and isinstance(self.pooling, tuple):
                    embedding = (hidden_states[self.pooling[0]] + hidden_states[self.pooling[1]]).mean(dim=1)
                else:
                    raise Exception("unknown pooling")

            embeddings.append(embedding[0])

        if (save):
            save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)

            embeddings = np.array(embeddings)
            embeddings = pd.concat([pd.DataFrame(dataset), pd.DataFrame(embeddings)], axis=1)

            # print(embeddings)
            # path = f'embeddings/{self.dataset_name}'
            # if self.strategy == "layer" and self.pooling == 32:
            #     path = f'{path}/{self.model_name}_{split}_embeddings.csv'
            # if self.strategy == "layer" and self.pooling != 32:
            #     path = f'{path}/{self.model_name}-{self.strategy}-{self.pooling}_{split}_embeddings.csv'
            # elif self.strategy == "range":
            #     path = f'{path}/{self.model_name}-{self.strategy}-{self.pooling}_{split}_embeddings.csv'
            # elif self.strategy == "pair":
            #     path = f'{path}/{self.model_name}-{self.strategy}-{self.pooling[0]}-and-{self.pooling[1]}_{split}_embeddings.csv'

            # embeddings.to_csv(path, sep='\t', index=False)

        return embeddings

# if __name__ == "__main__":
#     llama_embeddings = Llama_Embeddings("llama_7B","imdb")
            # 1) get embeddings using input ids
            # tokens = self.tokenizer(data_row['text'])
            # input_ids = tokens['input_ids']
            # using get_input_embedding method
            # with torch.no_grad():
            #     input_embeddings = self.model.get_input_embeddings()
            #     embedding = input_embeddings(
            #         torch.LongTensor([input_ids]))
            #     embedding = torch.mean(
            #         embedding[0], 0).cpu().detach()

            #     embeddings.append(embedding)

            # 2) get embeddings using other llm approaches