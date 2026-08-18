import torch
import numpy as np
import torch.nn as nn

Linear = nn.Linear(2, 3)
weights1 = Linear.weight
print (weights1)

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
weights2 = model.lm_head.weight
weights3 = model.transformer.h[0].attn.c_attn.weight
print(weights2.shape, weights3.shape)