
<h1>SentEval+</h1>

<p>We are introducing SentEval+ , a framework built based on <a href='https://github.com/facebookresearch/SentEval'>SentEval</a> to both generate and evaluate embeddings.</p>

First clone the project:
```bash
git clone https://github.com/nlp-lab-dr-lu/senteval-plus.git
```
Next, create a virtual environment to install dependencies and activate it:
```bash
cd senteval-plus
virtualenv env
source env/bin/activate
```
Then install requirments:
```bash
pip install -r requirements.txt
```

<h2>Where To Start</h2>
<p>You can use this repository to both generate and evaluate embeddings with different settings.
  <ol>
    <li> <a href='#eg'>Generate Embeddings</a> </li>
    <li> <a href='#ee'>Evaluation</a> </li>
  </ol>
</p>

<h2 id='eg'>Embedding Generation</h2>
<p>The entry point to generate embeddings is main.py file. A loop over encoders and datasets will generate embeddings in this file. You can chose between our provided data in data folder or add any dataset you want to evaluate. Make sure to follow our datasets format for next steps.</p>

<p>To generate embeddings you want for your data, open main.py and fill the two lists:</p> <br>
  
```python
datasets = ["mr", "cr", "subj", "mpqa", ... ] 
models = ["simcse", "bert", "all-mpnet-base-v2", "text-embedding-3-small", "llama-7B", ... ]
``` 
<p>By default, we support Bert-base-uncased, SimCSE, SBert(all-mpnet-base-v2), ChatGPT(text-embedding-3-small), AnglE-Bert, AnglE-LLaMA and LLaMA(2) as models and STS, MR, CR, SUBJ, TREC, MPQA, MRPC, and SSTF as datasets to generate embeddings. You can easily add your embeddings to our pipeline by creating a file in encoder directory and follow our encoder template to implement generate embedding code. Then you need to import your encoder in main.py and add it to the loop</p>

<p> You need to store your embeddings in a directory named `embeddings` in the root directory of the project. Take a look at two sample embeddings for MR and STS-B dataset in this direcotry to understand the structure of `embeddings` directory.</p>

<h2 id='ee'>Evaluate Embeddings</h2>
Once you have the embeddings you can evaluate them using our evaluate code. We support three tasks:
<ol>
  <li> <a href='#cls'>Classification</a> </li>
  <li> <a href='#sts'>Semantic Similarity</a> </li>
  <li> <a href='#clu'>Clustering</a> </li>
</ol> 

<h3 id='cls'>Classification</h3>
<p>To evaluate the embeddings with classification task you need to add your dataset and encoder names in a config dictionary and create the loop for evaluation. Note that you can modify both number of folds and classifier in this configuration. Your classifier options are Multi-Layer Perceptron (mlp), Random Forest (rf), and SVM (svm). However, to utilize GPU to speed up evaluation process you need to use `mlp` classifier. A sample configuration dictionary could be as follow:</p>

```python
encoders = ["bert", ...]
datasets = ["mr", ...]

EMBEDDINGS_PATH = 'embeddings/' # where you stored the embeddings
RESULTS_PATH = 'results/' # where you want to save the results of evaluation
config = {
    'EMBEDDINGS_PATH': EMBEDDINGS_PATH,
    'RESULTS_PATH': RESULTS_PATH,
    'classifier': 'mlp',
    'kfold': 5,
    'encoders': encoders,
    'datasets': datasets
}
```
<p>Then you can run the loop using our `Evaluation()` class:</p>

```python
from eval.eval_cls import Evaluation

eval = Evaluation(config)
eval.run()
```

<h3 id='sts'>Semantic Similarity</h3>
<p>Similar to classification task you need to set a config dictionary and run evaluation. </p>

```python
encoders = ["bert", ...]
datasets = ["stsb", ...]

EMBEDDINGS_PATH = 'embeddings/' # where you stored the embeddings
RESULTS_PATH = 'results/' # where you want to save the results of evaluation
config = {
    'EMBEDDINGS_PATH': EMBEDDINGS_PATH,
    'RESULTS_PATH': RESULTS_PATH,
    'encoders': encoders,
    'datasets': datasets
}
```
<p>Then you can run the loop using our `Evaluation()` class:</p>

```python
from eval.eval_sts import Evaluation

eval = Evaluation(config)
eval.run()
```

<h3 id='clu'>Clustering</h3>
<p>Our final evaluation task is clustering. You can run clustering using a similar confi structure to other tasks.</p>

```python
encoders = ["bert", ...]
datasets = ["mr", ...]

EMBEDDINGS_PATH = 'embeddings/' # where you stored the embeddings
SCORES_PATH = 'results/' # where you want to save the results of evaluation
config = {
    'EMBEDDINGS_PATH': EMBEDDINGS_PATH,
    'SCORES_PATH': RESULTS_PATH,
    'encoders': encoders,
    'datasets': datasets
}
```
<p>Then you can run the evaluation using our `ClusteringEvaluation()` class. Our clustering evaluation automatically, runs all whitening transformation on the embeddings as well.</p>

```python
from eval.clustering import ClusteringEvaluation

eval = ClusteringEvaluation(config)
eval.run()
```