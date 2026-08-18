# 简单的报告

## 第一次训练40次

- lr = 1e-3:
  - Loss: 4.6325
  - Loss: 23.8999
  - Loss: 9.2986
  - Loss: 25.8515
  - Loss: 10.3370
  - Loss: 6.4656

发现结果震荡明显，怀疑是lr过高导致

## 第二次我们调整学习率

- lr = 1e-4:
  - iter1, Loss: 4.2777
  - iter2, Loss: 3.6356
  - iter3, Loss: 3.3920
  - iter4, Loss: 3.2957
  - iter5, Loss: 3.1210
  - iter6, Loss: 2.9516
  - iter7, Loss: 2.8889
  - iter8, Loss: 2.8969
  - iter9, Loss: 2.8624
  - iter10, Loss: 2.7876

感觉这样就正常很多了
然后训练十次，尝试输出功能：

```text
What should I dotous fonnithis, beweeattheand;'rer, wU,
Aeathorengaremeco;fories's wisthingeathiXilothind the sureat blit borSin

 busR.istde'3eresonds ndI banore Osed wongrise cis thendihererenthethingy,
lll bersharUiend cotar
Ieacouseresthendsesthillll cesean, you s sar bUen e, t fn, cothis,
TooR:u hi,
Thinden, w,
veronthO,
Tg, bllR
T bke, nd,N

S.
-end hend cAto, hec, y,kyocpinisoc, tourererstthendANunouit mce,
An fner, ponghaThind hy ceR, t wourthathtoy the tMere, thar
A.
Towe!lenrse, ben, blparinwit id,
Ai
```

不知道在说什么
好的，我们现在一口气跑5000次训练
但是我发现跑的太慢了，决定减小一点模型的参数

## 训练模型，并且量化

- 然后我们训练1000次:
  - iter200, Loss: 2.4257
  - iter400, Loss: 2.1754
  - iter600, Loss: 1.8690
  - iter800, Loss: 1.7285
  - iter1000, Loss: 1.5787

最后我们跑了一下量化:
### 8-bit

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 5.5337 | / |
| absmax | 5.5309 | -0.0504% |
| absmax with per-channel quantization in channel 0 | 5.5343 | 0.0110% |
| absmax with per-channel quantization in channel 1 | 5.5354 | 0.0305% |
| zero-point | 5.5350 | 0.0231% |
| zero-point with per-channel quantization in channel 0 | 5.5326 | -0.0202% |
| zero-point with per-channel quantization in channel 1 | 5.5324 | -0.0231% |

* 发现，几乎没有什么差别，说明小模型的outlier是涌现出来的，这个参数的大小对于int8还是太轻松了，甚至有一些最后的困惑度比baseline还高，我认为是由于数据噪声导致的

### 7-bit

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 5.5478 | / |
| absmax | 5.5508 | 0.0542% |
| absmax with per-channel quantization in channel 0 | 5.5485 | 0.0124% |
| absmax with per-channel quantization in channel 1 | 5.5476 | -0.0033% |
| zero-point | 5.5493 | 0.0281% |
| zero-point with per-channel quantization in channel 0 | 5.5497 | 0.0353% |
| zero-point with per-channel quantization in channel 1 | 5.5466 | -0.0216% |

### 6-bit

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 5.6808 | / |
| absmax | 5.6778 | -0.0528% |
| absmax with per-channel quantization in channel 0 | 5.6830 | 0.0397% |
| absmax with per-channel quantization in channel 1 | 5.6860 | 0.0921% |
| zero-point | 5.6891 | 0.1470% |
| zero-point with per-channel quantization in channel 0 | 5.6789 | -0.0331% |
| zero-point with per-channel quantization in channel 1 | 5.6830 | 0.0388% |

- 现在开始出现一点误差了。但是也会出现一些不合理的情况，暂时认为int6还是可以轻松解决这个问题，我们继续往更低的位宽量化

### 5-bit

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 5.6039 | / |
| absmax | 5.6338 | 0.5330% |
| absmax with per-channel quantization in channel 0 | 5.6197 | 0.2821% |
| absmax with per-channel quantization in channel 1 | 5.6187 | 0.2644% |
| zero-point | 5.6275 | 0.4214% |
| zero-point with per-channel quantization in channel 0 | 5.6063 | 0.0435% |
| zero-point with per-channel quantization in channel 1 | 5.6220 | 0.3227% |

- 这个时候开始出现一些明显的误差和规律

### 4-bit

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 5.8209 | / |
| absmax | 5.9956 | 3.0015% |
| absmax with per-channel quantization in channel 0 | 5.8807 | 1.0269% |
| absmax with per-channel quantization in channel 1 | 5.8736 | 0.9050% |
| zero-point | 5.9402 | 2.0489% |
| zero-point with per-channel quantization in channel 0 | 5.8714 | 0.8670% |
| zero-point with per-channel quantization in channel 1 | 5.8681 | 0.8114% |

- 现在开始出现稳定的误差并且展示出一些规律了，首先就是zp的误差是总体要比abs的误差要低的。同时per-channel也是总体比直接对整体使用scale的误差更低
- 同时对dim = 0 进行perchannel计算的效果是相对比较好的

### 3-bit

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 5.7088 | / |
| absmax | 7.0629 | 23.7184% |
| absmax with per-channel quantization in channel 0 | 6.0817 | 6.5312% |
| absmax with per-channel quantization in channel 1 | 6.3420 | 11.0916% |
| zero-point | 6.3311 | 10.9011% |
| zero-point with per-channel quantization in channel 0 | 5.9868 | 4.8702% |
| zero-point with per-channel quantization in channel 1 | 5.9993 | 5.0887% |

- 随着转化的位宽减小，误差逐渐增大，并且我们上面总结的经验越来越明显

### 2-bit

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 5.5148 | / |
| absmax | 55.7119 | 910.2188% |
| absmax with per-channel quantization in channel 0 | 18.3996 | 233.6379% |
| absmax with per-channel quantization in channel 1 | 29.2681 | 430.7149% |
| zero-point | 18.7111 | 239.2861% |
| zero-point with per-channel quantization in channel 0 | 8.3557 | 51.5138% |
| zero-point with per-channel quantization in channel 1 | 10.3721 | 88.0759% |

- 到了int2简直是误差非常非常大了，同时上面的经验更加明显

* 在nn.Linear方法下dim = 0代表的是输出的通道，也是权重的输出通道的特征维度，说明输出通道的outlier是影响更大的,但是我们使用GPT2会得到相反的结果

## GPT2的量化结果

### 对于所有参数进行的量化

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 57.6772 | / |
| absmax | 67.7922 | 17.5373% |
| absmax with per-channel quantization in channel 0 | 61.319 | 6.3141% |
| absmax with per-channel quantization in channel 1 | 64.383 | 11.6266% |
| zero-point | 59.9295 | 3.905% |
| zero-point with per-channel quantization in channel 0 | 58.0924 | 0.7199% |
| zero-point with per-channel quantization in channel 1 | 61.217 | 6.1373% |

* GPT2也是channel 0 的时候差距最小，但是问题是GPT2的大部分参数所在的矩阵的输入通道和输出通道都是(in, out)的摆放，也就是Conv1D,这个问题跟我们的GPT是相反的，但是最后优化的结果却是相同的

首先，我们会怀疑我们自己做的模型的参数是不够的，导致outlier的涌现不够明显

**total weights: 85.2311 M , 124.3185 M**

但是我感觉实际上参数之间的差距也不是很多，为什么会出现这种貌似截然相反的结果呢？我觉得可以进行分层的量化，就是我们对于GPT2中的所有Conv1D层进行量化，但是不对Linear层量化，和相反的结果看看会发生什么

### 仅对Conv1D矩阵的参数进行量化

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 57.6772 | / |
| absmax | 62.9305 | 9.1081% |
| absmax with per-channel quantization in channel 0 | 57.1934 | -0.8388% |
| absmax with per-channel quantization in channel 1 | 57.4091 | -0.4648% |
| zero-point | 59.07 | 2.4148% |
| zero-point with per-channel quantization in channel 0 | 58.3241 | 1.1216% |
| zero-point with per-channel quantization in channel 1 | 58.05 | 0.6464% |

* 实际上我们可以发现，两个channel并没有多少的差别
所以我怀疑，会出现明显差别的可能只是Linear这个部分，也就是lm_head的部分

### 仅对Linear矩阵的参数进行量化

| 量化方式 | 困惑度 | 量化误差 |
| ------ | ------ | ------ |
| FP32 baseline | 57.6772 | / |
| absmax | 59.2826 | 2.7834% |
| absmax with per-channel quantization in channel 0 | 58.0181 | 0.591% |
| absmax with per-channel quantization in channel 1 | 60.4896 | 4.8761% |
| zero-point | 59.0323 | 2.3495% |
| zero-point with per-channel quantization in channel 0 | 57.7052 | 0.0485% |
| zero-point with per-channel quantization in channel 1 | 60.6983 | 5.2379% |

* 从上面的数据我们就可以看出，我们只修改Linear的矩阵（实际上就是lm_head层），这个层量化的误差是比Conv1D的层，也就是attn里面的c_attn 和 c_proj层更大的。
而且在这个层出现了非常明显的channel之间的区别dim = 1 的影响明显要比1大

那么对于同属于Linear的层和我的模型来说，这个结论就是完全相同的了，没有什么区别。

而Linear层的权重是(out, in)说明我们把输出通道压扁，最后(1, in)，对于不同的输入通道的特征维度，差别是比较明显的，outlier也会影响较大。

同时我们也发现了lm_head层相对于attention层是量化相对敏感的一层，这与TQS论文的发现：量化敏感性集中在输入输出模块，尤其是输出侧 的发现一致