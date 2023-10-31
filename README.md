# CleanTransformer

项目特点：

* 除了python基础库和pytorch基础运算外，不借助其他第三方库
* 从零推导、实现Transformer及Bert、GPT、Diffusion等热门模型
* 借助开源大模型权重，从零实现SFT+RLHF的训练、部署
* 从零实现data parallel、tensor parallel、pipeline parallel等并行训练策略

**欢迎大家来一起完善代码和教程**

文字教程见：

* [知乎: 从零实现BERT、GPT及Difussion类算法](https://zhuanlan.zhihu.com/p/624068993)

## 已更新
* [Tokenizer](https://zhuanlan.zhihu.com/p/624072556)
  * BPE原理、训练代码、分词代码
  * WordPiece原理、训练代码、分词代码
  * ULM原理
* [Multi-head Attention & Transformer](https://zhuanlan.zhihu.com/p/624343441)
  * Multi-Head Attention原理、代码实现
  * BatchNorm & LayerNorm原理、LayerNorm代码实现
  * TransformerBlock原理、代码实现
* [Bert & GPT1/2/3](https://zhuanlan.zhihu.com/p/625178027)
  * BertTokenizer原理、代码实现
  * BertModel、BertForSequenceClassification代码实现
  * GPTModel、GPTLMHeadModel代码实现
* [Greedy Search, Beam Search, Penalty, Sampling](https://zhuanlan.zhihu.com/p/629929349)
  * k_v_cache原理、代码实现
  * Batch Greedy Search原理、代码实现
  * Batch Beam Search原理、代码实现
  * Logits Penalty原理、代码实现
  * Logits Sampling原理、代码实现
* [模型训练MiniBloomChat: Bloom+SFT](https://zhuanlan.zhihu.com/p/635714662)
  * 模型选型，数据集选型，Bloom模型代码实现
  * MSELoss、NLLLoss 、CrossEntropyLoss原理、推导、代码实现
  * SGD、SGD+Momentum、SGD+Momentum+Weight Decay、AdaGrad、AdaDelta/RMSProp、AdamW原理、推导、代码实现
  * Dataset、DataLoader、TrainLoop代码实现
  * 流式生成、SFT后模型效果对比
* [自动混合精度](https://www.zhihu.com/question/306508382/answer/3253039603)
  * 自动混合精度O0，O1，O2，O3对比
  * 核心功能实现：cast_model_type、cast_model_outputs、cast_model_outputs、patch_torch_functions、patch_torch_functions
  * 扩展阅读：optimizer中的参数及方法

* [数据并行：DistributedDataParallel](https://www.zhihu.com/question/53851014/answer/3271802850)
  * 模型参数同步
  * grad的分组同步buckets构建、并行同步代码实现
  * 扩展阅读：
    * forward、backward计算过程详解
    * grad_fn、next_functions详解
    * register_hook详解


## 计划

- [x] 分词器Tokenizer: BPE, WordPiece
- [x] 原始Transformer: LayerNorm, Multi-Head Attention, TransformerLayer
- [x] 完整模型搭建及推理: Bert, GPT1/2/3, Bert Inference
- [x] 生成策略: Greedy Search, Beam Search, Logits Penalty, Logits Sampling
- [x] 模型训练: Optimizer (SGD, Adam), Loss (MSELoss, CrossEntropyLoss), Trainer, Deployment
- [x] 类ChatGPT模型训练: Bloom + SFT ~~+ RLHF~~
- [ ] （部分已完成）训练及推理加速: Data Parallel, Tensor Parallel, Pipeline Parallel, Activition Checkpoint, Model Quantization
- [ ] 文生图: Diffusion Model