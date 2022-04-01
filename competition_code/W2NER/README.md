 
# Unified Named Entity Recognition as Word-Word Relation Classification

论文代码仓库：https://github.com/ljynlp/W2NER

Source code for AAAI 2022 paper: [Unified Named Entity Recognition as Word-Word Relation Classification](https://arxiv.org/pdf/2112.10070.pdf)

### 商品实体识别
- process.py:将比赛数据集转换成w2ner的输入
- 设置配置文件，参考resume-zh.json直接复制了一份gaiic.json比赛数据集配置
- main.py：chinese-roberta-wwm-ext模型训练与评估
- main_nehza.py：nezha模型训练与评估
- convert_result.py:转换结果文件
- 线上：0.801


### Label Scheme
<p align="center">
  <img src="./figures/scheme.PNG" width="400"/>
</p>

### Architecture
<p align="center">
  <img src="./figures/architecture.PNG" />
</p>

## 1. Environments

```
- python (3.8.12)
- cuda (11.4)
```

## 2. Dependencies

```
- numpy (1.21.4)
- torch (1.10.0)
- gensim (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- prettytable (2.4.0)
```

## 3. Dataset

- [Conll 2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- [OntoNotes 4.0](https://catalog.ldc.upenn.edu/LDC2011T03)
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
- [ACE 2004](https://catalog.ldc.upenn.edu/LDC2005T09)
- [ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06)
- [GENIA](http://www.geniaproject.org/genia-corpus)
- [CADEC](https://pubmed.ncbi.nlm.nih.gov/25817970/)
- [ShARe13](https://clefehealth.imag.fr/?page_id=441)
- [ShARe14](https://sites.google.com/site/clefehealth2014/)

We provide some datasets processed in this [link](https://drive.google.com/drive/folders/1NdvUeIUUL3mlS8QwwnqM628gCK7_0yPv?usp=sharing).

## 4. Preparation

- Download dataset
- Process them to fit the same format as the example in `data/`
- Put the processed data into the directory `data/`

## 5. Training

```bash
>> python main.py --config ./config/example.json
```
## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 7. Citation

If you use this work or code, please kindly cite the following paper:

```
@article{li2021unified,
  title={Unified Named Entity Recognition as Word-Word Relation Classification},
  author={Li, Jingye and Fei, Hao and Liu, Jiang and Wu, Shengqiong and Zhang, Meishan and Teng, Chong and Ji, Donghong and Li, Fei},
  journal={arXiv preprint arXiv:2112.10070},
  year={2021}
}
```



