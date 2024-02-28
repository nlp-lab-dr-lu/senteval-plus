[![Open In Colab]([https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O_tqM3MnFIJl5NxoF9RSk4q_2DT6xLgV?usp=sharing](https://colab.research.google.com/notebooks/forms.ipynb))
<h1>SentEval+</h1>

<p>We are introducing SentEval+ , a framework built based on <a href='https://github.com/facebookresearch/SentEval'>SentEval</a> to both generate and evaluate embeddings.</p>

First clone the project:
```bash
git clone https://github.com/nlp-lab-dr-lu/senteval-plus.git
```
Then create a virtual environment to install dependencies and activate it:
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
<p>You can download the embeddings of x,y,z,... datasets from this table. If you need to add new dataset to evaluate:
  <ol>
    <li> Want to start from scratch? steps to <a href='#eg'>generate embeddings</a> using our code </li>
    <li> Already have the embeddings of your dataset? start <a href='#ee'>evaluation</a> </li>
    <li> Already evaluated? <a href='#p'>plot</a> your results using colab </li>
  </ol>
</p>

<h2 id='eg'>Embedding Generation</h2>
<p>The entry point to generate embeddings is main.py file. A loop over encoders and datasets will embeddings in this file. You can chose between our provided data in data folder or add any dataset you want to evaluate. Make sure to follow our datasets format for next steps.</p>

<p>To generate embeddings you want for your data, open main.py and fill the two lists:</p> <br>
  
```python
datasets = ["mr", "cr", "subj", "mpqa", ... ] 
models = ["simcse", "bert", "all-mpnet-base-v2", "text-embedding-3-small", "llama-7B", ... ]
``` 
<p>By default, we support Bert-base-uncased, SimCSE, SBert(all-mpnet-base-v2), ChatGPT(text-embedding-3-small), AnglE-Bert, AnglE-LLaMA and LLaMA(2) as models and STS, MR, CR, SUBJ, TREC, MPQA, MRPC as datasets to generate embeddings. You can easily add your embeddings to our pipeline by creating a file in encoder directory and follow our encoder template. Then you need to import your encoder in maon.py and add it to the loop</p>
<p> You need to store your embeddings in a directory named as `embeddings` in the root directory of the project.</p>
<!-- will add a bash file to automatically download embeddings from dr lu's website later -->
<h2 id='ee'>Evaluate Embeddings</h2>
Once you have the embeddings you can evaluate them using our evaluate code. We support two tasks: 1) Classification 2) Semantic Similarity
To evaluate the embeddings with classification task you need to add your dataset and encoder names in eval_cls.py file. This file automatically evaluate your embeddings and create a json file of the results of the 5-fold classification with an MLP in results folder. You can also create confusion matrix by changing the draw_cm value to True in the code.

<h2 id='p'>Plotting</h2>
To plot your results and compare them with each other you can use plots.ipynb notebook. You just need to add your data and encoder to the list of each plot and then generate the plots.
