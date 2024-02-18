from llama_base import Llama_Embeddings
from bert_base import Bert_Embeddings
from sbert_base import SBert_Embeddings
from angle_base import Angle_Embeddings
from chatGPT import ChatGPT_Embeddings
from simcse_base import SimCSE_Embeddings
from data import *

'''
options for models_bert:
    "bert"
'''
models_bert = ["bert"]
'''
options for models_sbert:
    all-MiniLM-L6-v2    : All-round model tuned for many use-cases. Trained on a large and diverse dataset of over 1 billion training pairs. D = 768
    all-mpnet-base-v2   : All-round model tuned for many use-cases. Trained on a large and diverse dataset of over 1 billion training pairs. D = 384
    all-distilroberta-v1: All-round model tuned for many use-cases. Trained on a large and diverse dataset of over 1 billion training pairs. D = 768
'''
models_sbert = ["all-mpnet-base-v2"]
'''
options for models_simcse:
    simcse
'''
models_simcse = ["simcse"]
'''
options for models_llama:
    "llama-7B", "llama-13B", "llama-30B", "llama-65B", "llama2-7B", "llama2-13B", "llama2-70B"
'''
models_llama = ["llama-7B", "llama2-7B"]
'''
options for models_angle:
    angle-bert    : fine tuned bert on nli dataset
    angle-llama   : fine tuned llama2 with lora technique on nli dataset
'''
models_angle = [ "angle-llama"]#"angle-bert",
'''
options for models_chatGPT:
    text-embedding-3-small  : 62.3% in MTEB, 62,500 pages per dollor
    text-embedding-3-large  : 64.6% in MTEB, 9,615 pages per dollor
    text-embedding-ada-002  : 61.0& in MTEB, 12,500 pages per dollor
'''
models_chatGPT = ["text-embedding-3-small"]
'''
options for datasets:
    built-in train/test split:
        "yelpp", "imdb", "agnews", "yelpf", "trec", "sstf", "mrpc1", "mrpc2"
'''
splitted_datasets = ["yelpp", "imdb", "agnews", "yelpf", "trec", "sstf", "mrpc"]
'''
    no built-in train/test split:
        "mr", "cr", "subj", "mpqa"
'''
unsplitted_datasets = ["mr", "cr", "subj", "mpqa"]
'''
    similarity tasks:
        "sts1", "sts2"
'''
similarity_datasets = ["sts1", "sts2"]

datasets = ["sst2"] 

models = models_llama #models_sbert + models_bert + models_simcse + models_angle + models_llama + models_chatGPT

for model in models:
    if(model in models_sbert):
        SBert_Embeddings(model, datasets)
    elif(model in models_bert):
        Bert_Embeddings(model, datasets)
    elif(model in models_simcse):
        SimCSE_Embeddings(model, datasets)
    elif(model in models_angle):
        Angle_Embeddings(model, datasets)
    elif(model in models_llama):
        Llama_Embeddings(model, datasets)
    elif(model in models_chatGPT):
        ChatGPT_Embeddings(model, datasets)

'''
options for bio datasets:
    "hiv", "bace", "clintox", "bbbp", "structure_links", "drug_discription"
'''
bio_datasets = ["hiv"]
# for model in models_llama:
#     Llama_Embeddings(model, bio_datasets)
