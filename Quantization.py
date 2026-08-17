import numpy as np
import math
import torch
import torch.nn as nn
import babyGPT
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
model = babyGPT.GPT().to(device)
model.load_state_dict(torch.load("BabyGPT-Quantization/checkpoint.pt"))

def absmax(idx):
    a = idx.abs().max()
    s = 127 / a
    y = torch.round(idx * s)
    return y, s

def absmax_perchannel0(idx):
    a = idx.abs().max(dim = 0)
    s = 127 / a
    y = torch.round(idx * s)
    return y, s

def absmax_perchannel1(idx):
    a = idx.abs().max(dim = 1)
    s = 127 / a
    y = torch,round(idx * s)
    return y, s

def zp(idx):
    a = idx.max()
    b = idx.min()
    s = 255/(a - b)
    z = torch.round(-s * b) - 128
    y = torch.round(s * idx + z)
    return y, s, z

def zp_perchannle0(idx):
    a = idx.max(dim = 0)
    b = idx.min(dim = 0)
    s = 255/(a - b)
    z = torch.round(-s * b) - 128
    y = torch.round(s * idx + z)
    return y, s, z

def zp_perchannel1(idx):
    a = idx.max(dim = 1)
    b = idx.min(dim = 1)
    s = 255/(a - b)
    z = torch.round(-s * b) - 128
    y = torch.round(s * idx + z)
    return y, s, z

def dequantize(y, s, z = None):
    return y / s if z is None else (y - z) / s

@torch.no_grad()
def cal_perplexity(model):
    x, y = babyGPT.get_batch("test")
    _, loss = model(x, y)
    return torch.exp(loss)

model_abs = copy.deepcopy(model)
for params in model_abs.parameters():
    y, s = absmax(params.data)
    params.data = dequantize(y, s)

model_abs_perchannel0 = copy.deepcopy(model)
for params in model_abs_perchannel0.parameters():
    y, s = absmax_perchannel0(params.data)
    params.data = dequantize(y, s)

model_abs_perchannel1 = copy.deepcopy(model)
for params in model_abs_perchannel1.parameters():
    y, s = absmax_perchannel1(params.data)
    params.data = dequantize(y, s)

model_zp = copy.deepcopy(model)
for params in model_zp.parameters():
    y, s, z = zp(params.data)
    params.data = dequantize(y, s, z)

model_zp_perchannel0 = copy.deepcopy(model)
for params in model_zp_perchannel0.parameters():
    y, s, z = zp_perchannle0(params.data)
    params.data = dequantize(y, s, z)

model_zp_perchannel1 = copy.deepcopy(model)
for params in model_zp_perchannel1.parameters():
    y, s, z = zp_perchannel1(params.data)
    params.data = dequantize(y, s, z)
