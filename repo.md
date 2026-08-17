第一次训练40次
lr = 1e-3
Loss: 4.6325
Loss: 23.8999
Loss: 9.2986
Loss: 25.8515
Loss: 10.3370
Loss: 6.4656
发现结果震荡明显，怀疑是lr过高导致
lr = 1e-4
iter1, Loss: 4.2777
iter2, Loss: 3.6356
iter3, Loss: 3.3920
iter4, Loss: 3.2957
iter5, Loss: 3.1210
iter6, Loss: 2.9516
iter7, Loss: 2.8889
iter8, Loss: 2.8969
iter9, Loss: 2.8624
iter10, Loss: 2.7876
感觉这样就正常很多了
然后训练十次，尝试输出功能：
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
不知道在说什么
好的，我们现在一口气跑5000次训练
但是我发现跑的太慢了，决定减小一点模型的参数
然后我们训练1000次
iter200, Loss: 2.4257
iter400, Loss: 2.1754
iter600, Loss: 1.8690
iter800, Loss: 1.7285
iter1000, Loss: 1.5787
最后我们跑了一下量化
tensor(5.5433, device='cuda:0')
tensor(5.5411, device='cuda:0')
tensor(5.5431, device='cuda:0')
tensor(5.5428, device='cuda:0')
tensor(5.5438, device='cuda:0')
tensor(5.5436, device='cuda:0')
tensor(5.5433, device='cuda:0')
发现，几乎没有什么差别，说明小模型的outlier是涌现出来的，这个参数的大小对于int8还是太轻松了
int6
tensor(5.7867, device='cuda:0')
tensor(5.7918, device='cuda:0')
tensor(5.7902, device='cuda:0')
tensor(5.7912, device='cuda:0')
tensor(5.7960, device='cuda:0')
tensor(5.7873, device='cuda:0')
tensor(5.7911, device='cuda:0')
int4
tensor(6.1025, device='cuda:0')
tensor(6.2620, device='cuda:0')
tensor(6.1657, device='cuda:0')
tensor(6.1712, device='cuda:0')
tensor(6.2569, device='cuda:0')
tensor(6.1273, device='cuda:0')
tensor(6.1663, device='cuda:0')
int3
ensor(5.6132, device='cuda:0')
tensor(6.9348, device='cuda:0')
tensor(5.9354, device='cuda:0')
tensor(6.1317, device='cuda:0')
tensor(6.2151, device='cuda:0')
tensor(5.8779, device='cuda:0')
tensor(5.8613, device='cuda:0')
int2
tensor(5.7363, device='cuda:0')
tensor(55.6390, device='cuda:0')
tensor(19.2894, device='cuda:0')
tensor(29.8066, device='cuda:0')
tensor(19.8817, device='cuda:0')
tensor(8.6657, device='cuda:0')
tensor(10.7952, device='cuda:0')
同时，我们也发现了per_channel 的时候对于dim = 0会显著减小