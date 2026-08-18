import numpy as np
import torch
import torch.nn as nn
import babyGPT

def absmax(x):
    assert x.dtype == torch.float32
    a = x.abs().max()
    s = (2 ** 7 - 1) / a
    #这里我们需要注意在python中没有^的幂运算，这个是XOR运算
    #幂运算是**
    y = torch.round(x * s)
    return y, s

def dequantize(x, s, z = None):
    return x / s if z is None else (x - z)/s

def asymmetric(x):
    assert x.dtype == torch.float32
    a, b = x.max(), x.min()
    s = (128 + 127) / (a - b)
    z = torch.round(-s * b) - 2 ** (7)
    y = torch.round(s * x + z)
    return y, s, z



# x1 = torch.randn(7, dtype = torch.float32)
# y1, s = absmax(x1)
# x2 = dequantize(y1, s)
# y2, s, z = asymmetric(x1)
# x3 = dequantize(y2, s, z)


# error1 = (x1 - x2).abs().mean()
# error2 = (x1 - x3).abs().mean()
# print (x1, y1, x2, error1)
# print (x1, y2, x3, error2)

#上面的这个部分是absmax 和 zero-point的算法的部分
#接下来我们来加载一下GPT2并且来量化一下他的权重
#有一些部分是进行验证
from transformers import AutoModelForCausalLM, AutoTokenizer
#我们现在加载因果语言模型的包和tokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
#我们现在首先加载一个模型，并且这个模型是已经训练完成的
#from_pretrained的意思就是加载预训练好的模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")
#这个和我们之前的char版的tokenizer是不一样的，它把文本切成字词，比我们之前写的encode/decode更加复杂，更加标准
# print (model, tokenizer)

total = sum(p.numel() for p in model.parameters() if p.data.ndim >= 2)
#我们这个代码的作用就是把全部的参数累加起来
#numel()函数是tensor的一个方法，作用是返回tensor元素的总数
#number of elements
print (f"一共有{total/1e6:.4f}M参数")
#124.4398M
# GPT2LMHeadModel(
#   (transformer): GPT2Model(
#     (wte): Embedding(50257, 768)
#     (wpe): Embedding(1024, 768)
#     (drop): Dropout(p=0.1, inplace=False)
#     (h): ModuleList(
#       (0-11): 12 x GPT2Block(
#         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
#这里我想着重说一下eps, elementwise_affine这两个参数，前面一个是防止除以0的小数
#y = (x - x.mean) / sqrt(方差 + eps)
#后面的elementwise_affine参数表示的是LayerNorm是否学习缩放和平移
#因为最后的输出 = 归一化的结果 * weight + bias， 如果要学习的话这个weight和bias都会进一步进行学习，能自动调整归一化之后的范围和位置
#         (attn): GPT2Attention(
#           (c_attn): Conv1D(nf=2304, nx=768)
#           (c_proj): Conv1D(nf=768, nx=768)
#           (attn_dropout): Dropout(p=0.1, inplace=False)
#           (resid_dropout): Dropout(p=0.1, inplace=False)
#         )
#         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
#         (mlp): GPT2MLP(
#           (c_fc): Conv1D(nf=3072, nx=768)
#           (c_proj): Conv1D(nf=768, nx=3072)
#           (act): NewGELUActivation()
#           (dropout): Dropout(p=0.1, inplace=False)
#         )
#       )
#     )
#     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
#   )
#   (lm_head): Linear(in_features=768, out_features=50257, bias=False)
#上面的是GPT2的大致的结构，我们需要拿出特定的参数的时候可以进行参考，实际上跟我们day1剖析的代码非常像，几乎就是一模一样
# w = model.transformer.h[0].attn.c_attn.weight.data
# y, s = absmax(w)
# w_back1 = dequantize(y, s)
# y, s, z = asymmetric(w)
# w_back2 = dequantize(y, s, z)
# error1 = (w - w_back1).abs().mean()
# error2 = (w - w_back2).abs().mean()
# print (error1, error2)
#上面是小试牛刀，对模型的一些部分进行量化，接下来我们就要开始量化整个模型，但是在我们量化整个模型之前，我们要先deepcopy一份，从而防止修改原来的模型
import copy
#这个是python自带的标准库，负责提供复制对象的功能，里面主要有两个函数分别是copy()函数进行浅拷贝和deepcopy()函数进行深拷贝
#深拷贝就是完全建立一个完全独立的克隆体。这么做是因为量化会修改原来的模型的权重
model_abs = copy.deepcopy(model)
for params in model_abs.parameters():
    y, s = absmax(params.data)
    #这里其实我有个困惑就是为什么我们要.data,那是因为param是parameter的子类，是带梯度跟踪的，并不是纯种的tensor
    params.data = dequantize(y, s)
#可以看到在这里我们直接修改了模型的参数
#我们再做一个
model_zp = copy.deepcopy(model)
for params in model_zp.parameters():
    y, s, z = asymmetric(params.data)
    params.data = dequantize(y, s, z)

def cal_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors = 'pt')
    #首先我们先来看一下这个代码的作用是什么，就是把text转换成token id 然后return_tensor = 'pt'表示最后以torch.tensor的形式返回
    #接下来我们来看一下tokenizer的内部结构：首先就是内部封装了一个BPE的字典tokenizer.encode转换为数字列表tokenizer.decode转换为文字
    #如果是直接调用的话就会返回一个字典，不仅仅是文字转换之后的数字列表，还会包含其他的东西
    #最后，我们需要的是张量，而不是一个列表
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels = input_ids)
        #错位已经封装在模型的内部了

    return round(torch.exp(outputs.loss).item(), 4)

def loss(x1, x2):
    return round((x2 - x1) / x1 * 100, 4)

with open("Transformer/my_essay/ass1.txt", "r", encoding = "utf-8") as f:
    test_text = f.read()[:2000]

perplexity1 = cal_perplexity(model, tokenizer, test_text)
perplexity2 = cal_perplexity(model_abs, tokenizer, test_text)
perplexity3 = cal_perplexity(model_zp, tokenizer, test_text)

#最后的结果
#tensor(57.6772) tensor(67.7922) tensor(59.9295)
#仔细研究上面的结果我们会发现最后的困惑度还是非常大的，而且absmax方法的困惑度最后的差值要远远大于zero-point方法的差值，我们怀疑是有特殊值outlier的存在
#接下来我们尝试一下per-channel量化，就是对每一行都进行计算，而不是一整个tensor都进行计算

def abs_channel1(x):
    a = x.abs().max(dim = -1, keepdim = True)[0]
    s = 127 / a
    y = torch.round(x * s)
    return y, s

def zero_point_channel1(x):
    a = x.max(dim = -1, keepdim = True)[0]
    b = x.min(dim = -1, keepdim = True)[0]
    s = 255 / (a - b)
    z = torch.round(-s * b) - 2 ** 7
    y = torch.round(x * s + z)
    return y, s, z

model_nabs1 = copy.deepcopy(model)
for params in model_nabs1.parameters():
    y, s = abs_channel1(params.data)
    params.data = dequantize(y, s)

model_nzp1 = copy.deepcopy(model)
for params in model_nzp1.parameters():
    y, s, z = zero_point_channel1(params.data)
    params.data = dequantize(y, s, z)

perplexity4 = cal_perplexity(model_nabs1, tokenizer, test_text)
perplexity5 = cal_perplexity(model_nzp1, tokenizer, test_text)

# tensor(57.6772) tensor(67.7922) tensor(59.9295)
# tensor(64.3831) tensor(61.2170)
#这里出现了问题，我们发现结果反而会更低

def abs_channel0(x):
    a = x.abs().max(dim = 0, keepdim = True)[0]
    s = 127 / a
    y = torch.round(x * s)
    return y, s

def zero_point_channel0(x):
    a = x.max(dim = 0, keepdim = True)[0]
    b = x.min(dim = 0, keepdim = True)[0]
    s = 255 / (a - b)
    z = torch.round(-s * b) - 2 ** 7
    y = torch.round(x * s + z)
    return y, s, z
#这里我们怀疑是维度的问题，所以我们更新一下维度继续跑
model_nabs0 = copy.deepcopy(model)
for params in model_nabs0.parameters():
    y, s = abs_channel0(params.data)
    params.data = dequantize(y, s)

model_nzp0 = copy.deepcopy(model)
for params in model_nzp0.parameters():
    y, s, z = zero_point_channel0(params.data)
    params.data = dequantize(y, s, z)

perplexity6 = cal_perplexity(model_nabs0, tokenizer, test_text)
perplexity7 = cal_perplexity(model_nzp0, tokenizer, test_text)

# tensor(57.6772) tensor(67.7922) tensor(59.9295)
# tensor(64.3831) tensor(61.2170)
# tensor(61.3190) tensor(58.0924)
#GPT2使用的是Conv1D所以权重的形状是(in, out)，我们求dim = 0最后的结果是(1, out),输入通道消失，dim = 1的最后的结果是(in, 1)，输出通道消失
#那么这样的结果分别是我们把scale放在输出通道和输入通道的结果，最后完全消除了输出通道和输入通道的outlier，结果是前者更加接近原来的模型，代表我们对于沿着输出通道分别进行scale的结果更好，也就是说输出通道相对独立，会更容易受到别的输出通道的影响，所以从整体上看，每一个输出通道上面的outlier对于整体权重矩阵的影响会更加大
#最后的结果果然显著减小了，所以可以看到outlier一般都是出现在权重的输出通道，但是为什么呢？
#接下来详细阅读一下《LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale》文章，阅读笔记在day5.md
#而且上面我们做的都是fake quantization，就是我们是进行计算精度的差距，但是没有实际上的进行量化

model_abs_Conv1D = copy.deepcopy(model)
for params in model_abs_Conv1D.transformer.h.parameters():
    y, s = absmax(params.data)
    params.data = dequantize(y, s)

model_zp_Conv1D = copy.deepcopy(model)
for params in model_zp_Conv1D.transformer.h.parameters():
    y, s, z = asymmetric(params.data)
    params.data = dequantize(y, s, z)

model_nabs1_Conv1D = copy.deepcopy(model)
for params in model_nabs1_Conv1D.transformer.h.parameters():
    y, s = abs_channel1(params.data)
    params.data = dequantize(y, s)

model_nzp1_Conv1D = copy.deepcopy(model)
for params in model_nzp1_Conv1D.transformer.h.parameters():
    y, s, z = zero_point_channel1(params.data)
    params.data = dequantize(y, s, z)

model_nabs0_Conv1D = copy.deepcopy(model)
for params in model_nabs0_Conv1D.transformer.h.parameters():
    y, s = abs_channel0(params.data)
    params.data = dequantize(y, s)

model_nzp0_Conv1D = copy.deepcopy(model)
for params in model_nzp0_Conv1D.transformer.h.parameters():
    y, s, z = zero_point_channel0(params.data)
    params.data = dequantize(y, s, z)

perplexity8 = cal_perplexity(model_abs_Conv1D, tokenizer, test_text)
perplexity9 = cal_perplexity(model_nabs0_Conv1D, tokenizer, test_text)
perplexity10 = cal_perplexity(model_nabs1_Conv1D, tokenizer, test_text)
perplexity11 = cal_perplexity(model_zp_Conv1D, tokenizer, test_text)
perplexity12 = cal_perplexity(model_nzp0_Conv1D, tokenizer, test_text)
perplexity13 = cal_perplexity(model_nzp1_Conv1D, tokenizer, test_text)


model_abs_Linear = copy.deepcopy(model)
for params in model_abs_Linear.lm_head.parameters():
    y, s = absmax(params.data)
    params.data = dequantize(y, s)

model_zp_Linear = copy.deepcopy(model)
for params in model_zp_Linear.lm_head.parameters():
    y, s, z = asymmetric(params.data)
    params.data = dequantize(y, s, z)

model_nabs1_Linear = copy.deepcopy(model)
for params in model_nabs1_Linear.lm_head.parameters():
    y, s = abs_channel1(params.data)
    params.data = dequantize(y, s)

model_nzp1_Linear = copy.deepcopy(model)
for params in model_nzp1_Linear.lm_head.parameters():
    y, s, z = zero_point_channel1(params.data)
    params.data = dequantize(y, s, z)

model_nabs0_Linear = copy.deepcopy(model)
for params in model_nabs0_Linear.lm_head.parameters():
    y, s = abs_channel0(params.data)
    params.data = dequantize(y, s)

model_nzp0_Linear = copy.deepcopy(model)
for params in model_nzp0_Linear.lm_head.parameters():
    y, s, z = zero_point_channel0(params.data)
    params.data = dequantize(y, s, z)

perplexity14 = cal_perplexity(model_abs_Linear, tokenizer, test_text)
perplexity15 = cal_perplexity(model_nabs0_Linear, tokenizer, test_text)
perplexity16 = cal_perplexity(model_nabs1_Linear, tokenizer, test_text)
perplexity17 = cal_perplexity(model_zp_Linear, tokenizer, test_text)
perplexity18 = cal_perplexity(model_nzp0_Linear, tokenizer, test_text)
perplexity19 = cal_perplexity(model_nzp1_Linear, tokenizer, test_text)

print (perplexity1)
print (perplexity2, loss(perplexity1, perplexity2))
print (perplexity3, loss(perplexity1, perplexity3))
print (perplexity4, loss(perplexity1, perplexity4))
print (perplexity5, loss(perplexity1, perplexity5))
print (perplexity6, loss(perplexity1, perplexity6))
print (perplexity7, loss(perplexity1, perplexity7))
print (perplexity8, loss(perplexity1, perplexity8))
print (perplexity9, loss(perplexity1, perplexity9))
print (perplexity10, loss(perplexity1, perplexity10))
print (perplexity11, loss(perplexity1, perplexity11))
print (perplexity12, loss(perplexity1, perplexity12))
print (perplexity13, loss(perplexity1, perplexity13))
print (perplexity14, loss(perplexity1, perplexity14))
print (perplexity15, loss(perplexity1, perplexity15))
print (perplexity16, loss(perplexity1, perplexity16))
print (perplexity17, loss(perplexity1, perplexity17))
print (perplexity18, loss(perplexity1, perplexity18))
print (perplexity19, loss(perplexity1, perplexity19))