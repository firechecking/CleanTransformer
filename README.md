# CleanTransformer

项目特点：

* 除了python基础库和pytorch基础运算外，不借助其他第三方库
* 从零推导、实现Transformer及Bert、GPT、Diffusion等热门模型
* 从零实现data parallel、tensor parallel、pipeline parallel等并行训练策略
* 借助开源大模型权重，从零实现SFT+RLHF的训练、部署

**平时工作忙，如果有同学有精力和意愿来一起完善代码和教程，欢迎私信联系**

文字教程见：

* [知乎: 从零实现BERT、GPT及Difussion类算法](https://zhuanlan.zhihu.com/p/624068993)
* [B站: 从零实现BERT、GPT及Difussion类算法](https://www.bilibili.com/read/cv23237718)

## 计划

- [x] 分词器Tokenizer: BPE, WordPiece
- [x] 原始Transformer: LayerNorm, Multi-Head Attention, TransformerLayer
- [x] 完整模型搭建及推理: Bert, GPT1/2/3, Bert Inference
- [ ] 生成策略: Greedy Search, Beam Search
- [ ] 模型训练: Optimizer (SGD, Adam), Loss (MSELoss, CrossEntropyLoss), Trainer, Deployment
- [ ] 类ChatGPT模型训练: LLaMA + SFT + RLHF
- [ ] 训练及推理加速: Data Parallel, Tensor Parallel, Pipeline Parallel, Activition Checkpoint, Model Quantization
- [ ] 文生图: Diffusion Model