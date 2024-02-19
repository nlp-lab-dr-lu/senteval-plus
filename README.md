<h1>SentEval+</h1>
<p>We are introducing SentEval+ , a framework built based on <a href='https://github.com/facebookresearch/SentEval'>SentEval</a> to both generate and evaluate embeddings.</p>

First clone the project:
```bash
git clone https://github.com/FlockOfBird/emb-eval.git
```
<h2>Where To Start</h2>
<p>You can download the embeddings of x,y,z,... datasets from this table. If you need to add new dataset to evaluate:
  <li> 1) Want to start from scratch? steps to <a href='#eg'>generate embeddings</a> using our code </li>
  <li> 2) Already have the embeddings of your dataset? start <a href='#ee'>evaluation</a> </li>
  <li> 3) Want to <a href='#ru'>repeat</a> our evaulations? follow these steps. </li>
  <li> 4) Already evaluated? <a href='#p'>plot</a> your results using colab </li>
</p>

<h2 id='eg'>Embedding Generation</h2>
The entry point to generate embeddings is main.py file. You can loop over encoders and datasets to generate embeddings here. You can chose between our provided data in data folder or add any dataset you want to evaluate. Make sure to follow our datasets format for next steps.
<h2 id='ee'>Evaluate Embeddings</h2>

<h2 id='ru'>Repeat Us</h2>

<h2 id='p'>Plotting</h2>
