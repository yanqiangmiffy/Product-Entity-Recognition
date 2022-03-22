# 中文命名实体识别
BiLSTM-CRF 和 Transformer-CRF 模型的 Pytorch 实现，以模块化的方式实现 CRF 以支持直接应用于 BiLSTM 和 Transformer 模型的输出

### 数据集
CLUENER 中文细粒度命名实体识别数据集，详细的介绍见 [CLUE](https://www.cluebenchmarks.com/introduce.html)；在当前目录新建`cluener`文件夹并下载数据集。

### 使用
```bash
python main.py --arch transformer
```
