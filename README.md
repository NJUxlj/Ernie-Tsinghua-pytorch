# Ernie-Tsinghua-pytorch


## Background


## Model Structure

## Main contributions

### 1. Entity Alignment
- 将文本中提到的实体与知识图谱中的实体进行对齐
- 使用特殊的对齐算法确保准确匹配
- 建立文本空间和知识空间的桥梁


### 2. Knowledge Integration
- 设计了**聚合器（aggregator）**来融合实体信息
- 实现了文本表示和知识表示的双向互补
- 通过**多头注意力机制**实现细粒度的知识融合

### 3. Pre-training Task
ERNIE采用了多个预训练任务：

#### 1. 掩码语言模型（MLM）：
类似BERT的掩码预测
增强模型的语言理解能力

#### 2. 下一句预测（NSP）：
预测两个句子是否连续
提升篇章级理解能力

#### 3. 知识掩码预测（dEA）：
预测被掩码的实体
强化模型对知识的理解和利用



### Experiment Results
ERNIE在多个任务上都取得了显著提升

1. 关系分类：
- 提升了8.4%的准确率
- 显著优于基准模型

2. 实体分类：
- 准确率提升3.4%
- 展现出强大的实体理解能力

3. 问答任务：
- 在多个数据集上都有明显提升
- 特别是在需要知识推理的问题上表现突出

## project structure
```Plain Text

ernie/  
├── config/  
│   └── model_config.py  
├── data/  
│   ├── dataset.py  
│   └── data_collator.py  
├── models/  
│   ├── embeddings.py  
│   ├── encoder.py    
│   └── ernie.py  
├── trainers/  
│   └── pretrain_trainer.py  
├── utils/  
│   ├── common.py  
│   └── metrics.py  
└── main.py


```







## Citation
```bibtex


```
