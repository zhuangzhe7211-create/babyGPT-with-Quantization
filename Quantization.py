import numpy as np
import math
import torch
import torch.nn as nn
import babyGPT
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
model = babyGPT.GPT().to(device)
model.load_state_dict(torch.load("BabyGPT-Quantization/checkpoint.pt"))
byte = 5
total = sum(p.numel() for p in model.parameters() if p.data.ndim >= 2)
print (f"total weights: {total/1e6:.4f} M")
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
    return torch.exp(loss).item()

def loss(x1, x2):
    return (x2 - x1) / x1 * 100

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
loss1 = loss(per_baseline, per_abs)
loss2 = loss(per_baseline, per_abs_perchannel0)
loss3 = loss(per_baseline, per_abs_perchannel1)
loss4 = loss(per_baseline, per_zp)
loss5 = loss(per_baseline, per_zp_perchannel0)
loss6 = loss(per_baseline, per_zp_perchannel1)

print(f"FP32 baseline: {per_baseline:.4f}")
print(f"absmax: {per_abs:.4f}, loss: {loss1:.4f}%")
print(f"absmax with per-channel quantization in channel 0: {per_abs_perchannel0:.4f}, loss: {loss2:.4f}%")
print(f"absmax with per-channel quantization in channel 1: {per_abs_perchannel1:.4f}, loss: {loss3:.4f}%")
print(f"zero-point: {per_zp:.4f}, loss: {loss4:.4f}%")
print(f"zero-point with per-channel quantization in channel 0: {per_zp_perchannel0:.4f}, loss: {loss5:.4f}%")
print(f"zero-point with per-channel quantization in channel 1: {per_zp_perchannel1:.4f}, loss: {loss6:.4f}%")
# tensor(5.6891, device='cuda:0')
# tensor(5.6880, device='cuda:0')
# tensor(5.6898, device='cuda:0')
# tensor(5.6896, device='cuda:0')
# tensor(5.6900, device='cuda:0')
# tensor(5.6896, device='cuda:0')
# tensor(5.6886, device='cuda:0')