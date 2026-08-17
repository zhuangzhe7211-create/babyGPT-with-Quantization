import numpy as np
import math
import torch
import torch.nn as nn
import babyGPT
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
model = babyGPT.GPT().to(device)
model.load_state_dict(torch.load("BabyGPT-Quantization/checkpoint.pt"))
byte = 2
def absmax(idx, bytes = 8):
    a = idx.abs().max()
    s = (2 ** (bytes - 1) - 1) / a
    y = torch.round(idx * s)
    return y, s

def absmax_perchannel0(idx, bytes = 8):
    a = idx.abs().max(dim = 0, keepdim = True)[0]
    s = (2 ** (bytes - 1) - 1) / a
    y = torch.round(idx * s)
    return y, s

def absmax_perchannel1(idx, bytes = 8):
    a = idx.abs().max(dim = 1, keepdim = True)[0]
    s = (2 ** (bytes - 1) - 1) / a
    y = torch.round(idx * s)
    return y, s

def zp(idx, bytes = 8):
    a = idx.max()
    b = idx.min()
    s = (2 ** bytes - 1)/(a - b)
    z = torch.round(-s * b) - (2 ** (bytes - 1))
    y = torch.round(s * idx + z)
    return y, s, z

def zp_perchannle0(idx, bytes = 8):
    a = idx.max(dim = 0, keepdim = True)[0]
    b = idx.min(dim = 0, keepdim = True)[0]
    s = (2 ** bytes - 1)/(a - b)
    z = torch.round(-s * b) - (2 ** (bytes - 1))
    y = torch.round(s * idx + z)
    return y, s, z

def zp_perchannel1(idx, bytes = 8):
    a = idx.max(dim = 1, keepdim = True)[0]
    b = idx.min(dim = 1, keepdim = True)[0]
    s = (2 ** bytes - 1)/(a - b)
    z = torch.round(-s * b) - (2 ** (bytes - 1))
    y = torch.round(s * idx + z)
    return y, s, z

def dequantize(y, s, z = None):
    return y / s if z is None else (y - z) / s

x_test, y_test = babyGPT.get_batch("test")
@torch.no_grad()
def cal_perplexity(model, x, y):
    _, loss = model(x, y)
    return torch.exp(loss)

model_abs = copy.deepcopy(model)
for params in model_abs.parameters():
    if params.data.ndim >= 2:
        y, s = absmax(params.data, byte)
        params.data = dequantize(y, s)

model_abs_perchannel0 = copy.deepcopy(model)
for params in model_abs_perchannel0.parameters():
    if params.data.ndim >= 2:
        y, s = absmax_perchannel0(params.data, byte)
        params.data = dequantize(y, s)

model_abs_perchannel1 = copy.deepcopy(model)
for params in model_abs_perchannel1.parameters():
    if params.data.ndim >= 2:
        y, s = absmax_perchannel1(params.data, byte)
        params.data = dequantize(y, s)

model_zp = copy.deepcopy(model)
for params in model_zp.parameters():
    if params.data.ndim >= 2:
        y, s, z = zp(params.data, byte)
        params.data = dequantize(y, s, z)

model_zp_perchannel0 = copy.deepcopy(model)
for params in model_zp_perchannel0.parameters():
    if params.data.ndim >= 2:
        y, s, z = zp_perchannle0(params.data, byte)
        params.data = dequantize(y, s, z)

model_zp_perchannel1 = copy.deepcopy(model)
for params in model_zp_perchannel1.parameters():
    if params.data.ndim >= 2:
        y, s, z = zp_perchannel1(params.data, byte)
        params.data = dequantize(y, s, z)

per_baseline = cal_perplexity(model, x_test, y_test)
per_abs = cal_perplexity(model_abs, x_test, y_test)
per_abs_perchannel0 = cal_perplexity(model_abs_perchannel0, x_test, y_test)
per_abs_perchannel1 = cal_perplexity(model_abs_perchannel1, x_test, y_test)
per_zp = cal_perplexity(model_zp, x_test, y_test)
per_zp_perchannel0 = cal_perplexity(model_zp_perchannel0, x_test, y_test)
per_zp_perchannel1 = cal_perplexity(model_zp_perchannel1, x_test, y_test)

print(per_baseline)
print(per_abs)
print(per_abs_perchannel0)
print(per_abs_perchannel1)
print(per_zp)
print(per_zp_perchannel0)
print(per_zp_perchannel1)
# tensor(5.6891, device='cuda:0')
# tensor(5.6880, device='cuda:0')
# tensor(5.6898, device='cuda:0')
# tensor(5.6896, device='cuda:0')
# tensor(5.6900, device='cuda:0')
# tensor(5.6896, device='cuda:0')
# tensor(5.6886, device='cuda:0')