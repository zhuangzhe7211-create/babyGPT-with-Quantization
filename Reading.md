# 阅读笔记

## 《Quantization sensitivity concentrates in input-output projection modules》

基于轨迹的量化敏感性分析（TQS）

训练后量化处理可以(Post-training quantization PTQ)可以在训练之后将全精度权重以及某些激活函数替换为低精度表示，因此来减小内存需求

### 论文核心贡献：

1. TQS，一种全新的量化敏感性评分方法
2. 基于动态系统的量化敏感性分析
3. TQS-PTQ, 一种无需校准的混合精度分配方法，适用于连续压缩预算
4. 研究表明，对于LLM，量化敏感性反而集中在输入/输出投影模块中
5. TQS敏感性排序方法，只需要进行一次敏感性测试

### Quantization sensitivity concentrates in input-output projection modules

量化灵敏度主要集中在输入输出的边界