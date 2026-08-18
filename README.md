# babyGPT-with-Quantization

从零训练一个GPT，进行简单的量化操作，进行学习。

## 具体做了什么

- 从零训练一个字符级GPT（GPT-2 的架构，85M参数，莎士比亚语料）
- 手写六种量化方法（absmax，zero-point，并且分别沿着输入和输出维度进行per-channel quantization）
- 进行隔离实验，研究手写的GPT与GPT2进行量化上的差别

## 核心发现

- 小模型几乎没有涌现出outlier，并且对于参数较小的模型，int8几乎无损
- 量化在 4-bit 以下的时候突然崩溃，误差极高
- zero-point + per_channel的方法量化效果在误差较高的位宽下展示了更强的性能
- GPT2的lm_head 层相比于attn内部的c_attn, c_proj层是更加量化敏感的层，这与TQS论文对于LLM的输入和输出更加量化敏感结论现象相近
- 对于lm_head 和我的模型，我们沿着输出通道量化（dim = 0）效果要明显好于输入通道（dim = 1），但是最后的结果其实是我们将输入通道的压扁，可以认为在不同的输入通道，权重的差异较大，所以量化结果会更好

## 关键结果

markdown repo.md

## 复现

python babyGPT.py

python Quantization.py

python Quantization-GPT2.py