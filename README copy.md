## Always monitor the GPUs while using delta, matrix
you can monitor the system using these commands:

```bash
$ nvidia-smi
$ glances
$ htop
```
check GPU status
```python
import torch

print('if cuda is available:', torch.cuda.is_available())
print('current cuda device:', torch.cuda.current_device())
print('number of cuda devices', torch.cuda.device_count())
```
## Prepare enviornment
Before running any python file, first active virtual environment with
```bash
$ source env/bin/activate
```
You can also install requirements if want to run the code on a different server. First freeze the requirements and then copy the requirements.txt file to your destination server and run `pip install -r requirements.txt`. Note that converted weights must be reconverted if you are changing the directories or servers. NERVER change the path of *B_converted directories on this server.

## Flags (Arguments)
You can plot embeddings if you pass `--Plot True` argument. for example:
```bash
$ python bert_base.py --Plot True
```
___
## Deal with huge models
### See layers
This is an example of how you can see the layers of your model:
```python
device = {"block1": 0, "block2": 1, "block3": 2, "block4": 3 }

config = LlamaConfig.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
with init_empty_weights():
    self.model = LlamaForCausalLM._from_config(config)

device = infer_auto_device_map(self.model)
print(device)
```

### LLaMA 30B
To run llama-30b you need to change the code for loading model to this:
```python
device = "auto"
self.model = LlamaForCausalLM.from_pretrained(
    PATH_TO_CONVERTED_WEIGHTS,
    device_map=device,
    max_memory={0: "12GiB", 1: "12GiB", 2:"12GiB", 3:"12GiB"},
    offload_folder="offload"
)
```
Since a lot of layers are going to load in cpu in this approach, the loading time of the model is very high (nearly 45 minutes).
### Other LLaMAs
```python
device = "auto"
self.model = LlamaForCausalLM.from_pretrained(
    PATH_TO_CONVERTED_WEIGHTS,
    device_map=device
)
```

### LLaMA weights
```python
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 2048
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: ftype      = 1 (mostly F16)
llama_model_load_internal: n_ff       = 11008
llama_model_load_internal: n_parts    = 1
llama_model_load_internal: model size = 7B
llama_model_load_internal: mem required  = 14645.07 MB (+ 1026.00 MB per state)
llama_init_from_file: kv self size  =  256.00 MB
```

### LLaMA Layers
iterate over the layers of a model and also want to know their names
```python
for name, layer in self.model.named_children():
  print(name, layer)
model LlamaModel(
  (embed_tokens): Embedding(32000, 4096, padding_idx=0)
  (layers): ModuleList(
    (0-31): 32 x LlamaDecoderLayer(
      (self_attn): LlamaAttention(
        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
        (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
        (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
        (act_fn): SiLUActivation()
      )
      (input_layernorm): LlamaRMSNorm()
      (post_attention_layernorm): LlamaRMSNorm()
    )
  )
  (norm): LlamaRMSNorm()
)
lm_head Linear(in_features=4096, out_features=32000, bias=False)
```