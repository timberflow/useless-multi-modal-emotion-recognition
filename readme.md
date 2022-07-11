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
|-- ResNet.py # 没用
|-- VGG16.py # 没用
```

## 运行流程
直接运行main.ipynb里的每个单元格即可

消融模型可以直接修改输入，包括训练里和所有的预测里都改一下。
```python
logits = model(img=img,src_ids=source_ids,src_mask=source_mask)
```
例如只使用图像数据：
```python
logits = model(img=img)
```
只使用text数据
```python
logits = model(src_ids=source_ids,src_mask=source_mask)
```

## 模型结构

![hh](https://github.com/timberflow/useless-multi-modal-emotion-recognition/blob/main/structure.png)






