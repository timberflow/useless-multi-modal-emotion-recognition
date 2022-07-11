# 多模态情感分析

没啥意义

## 依赖

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.10.0

- torch-geometric==2.0.2

- networkx==2.3

- scipy==1.5.4

- numpy==1.19.2

- sklearn==0.0

- matplotlib==3.1.1

- pandas==1.1.5


## 项目结构
We select some important files for detailed description.

```python
|-- data/
    |-- data/
    |-- train.txt
    |-- test_without_label.txt
|-- pretrained_model/
    |-- roberta-base # 太大了可能传不上
|-- main.ipynb # 在这里面运行每个单元格即可
|-- GoogLeNet.py # 如题
|-- Combination.py # 融合模型和所有的其它模型
```

## 运行流程
直接运行main.ipynb里的每个单元格即可

## 模型结构

![hh](https://github.com/timberflow/useless-multi-modal-emotion-recognition/blob/main/structure.png)






