from tqdm import tqdm
import torch
import numpy as np
from torch.nn import DataParallel
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import emb_util
from emb_util import logger, save_embeddings
from data import get_dataset
from angle_emb import AnglE, Prompts

class Angle_Embeddings:
    def __init__(self, model_name, datasets):

        self.model_name = model_name
        self.datasets = datasets

        for dataset in self.datasets:
            logger.info(f'encoding {dataset} dataset with {self.model_name} model')
            self.dataset_name = dataset
            dataset = get_dataset(dataset) 
            self.train_data, self.test_data = dataset["train"], dataset["test"]
            if(self.dataset_name in emb_util.splitted_datasets or self.dataset_name in emb_util.bio_datasets):
                train_embeddings = self.get_embeddings(self.train_data, self.dataset_name+'_train', self.model_name, True)
                test_embeddings = self.get_embeddings(self.test_data, self.dataset_name+'_test', self.model_name, True)
            else:
                embeddings = self.get_embeddings(self.train_data, self.dataset_name, self.model_name, True)

    def get_embeddings(self, dataset, split, angle_type, save):
        embeddings = []
        batch_size = 32
        if(angle_type=='angle-bert'):
            model_id = 'SeanLee97/angle-bert-base-uncased-nli-en-v1'
            model = AutoModel.from_pretrained(model_id).cuda()
            model = DataParallel(model).cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            for i in tqdm(range(0, len(dataset), batch_size)):
                batch_texts = dataset['text'][i:i + batch_size].tolist()
                batch_texts = ["" if isinstance(x, float) and np.isnan(x) else x for x in batch_texts]
                tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                tokens = {k: v.cuda() for k, v in tokens.items()}  # Move tokens to GPU
                with torch.no_grad():  # Disable gradient calculation for inference
                    outputs = model(**tokens)
                    embeddings_batch = torch.mean(outputs.last_hidden_state, dim=1)
                    embeddings.append(embeddings_batch)
            embeddings = torch.cat(embeddings, dim=0)
            embeddings = embeddings.cpu().numpy()

        elif(angle_type=='angle-llama'):
            device = "auto"
            model_id = 'SeanLee97/angle-llama-7b-nli-v2'
            config = PeftConfig.from_pretrained(model_id)
            base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map=device)
            model = PeftModel.from_pretrained(base_model, model_id, device_map=device)            
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            def decorate_text(text: str):
                return Prompts.A.format(text=text)
            for i in tqdm(range(0, len(dataset), batch_size)):
                batch_texts = dataset['text'][i:i + batch_size].tolist()
                batch_texts = ["" if isinstance(x, float) and np.isnan(x) else x for x in batch_texts]
                for i, text in enumerate(batch_texts):
                    batch_texts[i] = decorate_text(text)
                tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
                tokens = {k: v.cuda() for k, v in tokens.items()}
                with torch.no_grad():  
                    outputs = model(output_hidden_states=True, **tokens).hidden_states[-1][:, -1].float().detach().cpu().numpy()
                    embeddings.extend(outputs)

        if (save):
            save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)

    # def get_embeddings2(self, dataset, split, angle_type, save):

    #     '''
    #         dataset: a hugging face dataset object
    #         split: the dataset name string to save embeddings in a path
    #         save: T/F value to save the model in results directory
    #         type: angle-bert or angle-llama
    #     '''
    #     if(angle_type=='angle-bert'):
    #         angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg', device="cuda:1").cuda()
    #     elif(angle_type=='angle-llama'):
    #         angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf', pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2').cuda()
        
    #     angle.set_prompt(prompt=Prompts.A)
        
    #     embeddings = []
    #     for i, data_row in tqdm(dataset.iterrows()):
    #         data_row['text'] = '' if data_row['text'] == None else data_row['text']
    #         embedding = angle.encode([{'text': data_row['text']}], to_numpy=True)
    #         embeddings.append(embedding[0])
        
    #     if (save):
    #         emb_util.save_embeddings(embeddings, dataset, self.model_name, self.dataset_name, split)

    #     return embeddings