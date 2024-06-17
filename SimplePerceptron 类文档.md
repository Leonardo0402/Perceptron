### SimplePerceptron 类文档

#### 简介

`SimplePerceptron` 类实现了一个单层感知机模型，包括批量更新、正则化、动态学习率和早停机制。此类还包含标准化和归一化数据的方法。

#### 初始化方法

```python
def __init__(self, input_dim, l_rate=0.01, max_iter=1000, batch_size=10, lambda_reg=0.01, decay=0.99, early_stopping_rounds=10):
```

- `input_dim` (int): 输入特征的维度。
- `l_rate` (float): 学习率，默认为 0.01。
- `max_iter` (int): 最大迭代次数，默认为 1000。
- `batch_size` (int): 批量大小，默认为 10。
- `lambda_reg` (float): 正则化参数，默认为 0.01。
- `decay` (float): 学习率衰减，默认为 0.99。
- `early_stopping_rounds` (int): 早停机制的轮数，默认为 10。

#### 激活函数

```python
def activation(self, x):
```

根据输入 `x` 和权重计算加权和，并使用符号函数（sign function）作为激活函数返回分类结果。

#### 训练方法

```python
def fit(self, X_train, y_train, X_val=None, y_val=None):
```

- `X_train` (list of list of floats): 训练集特征数据。
- `y_train` (list of ints): 训练集标签数据。
- `X_val` (list of list of floats, optional): 验证集特征数据，默认为 None。
- `y_val` (list of ints, optional): 验证集标签数据，默认为 None。

训练感知机模型，支持批量更新、正则化、动态学习率和早停机制。

返回值:

- `str`: 表示模型训练完成的字符串。

#### 预测方法

```python
def predict(self, X):
```

- `X` (list of list of floats): 输入特征数据。

返回值:

- `list of ints`: 预测的标签。

#### 标准化方法

```python
@staticmethod
def standardize(X):
```

- `X` (list of list of floats): 输入特征数据。

返回值:

- `list of list of floats`: 标准化后的特征数据。

#### 归一化方法

```python
@staticmethod
def normalize(X):
```

- `X` (list of list of floats): 输入特征数据。

返回值:

- `list of list of floats`: 归一化后的特征数据。

### 交叉验证函数文档

#### 简介

`cross_validate` 函数使用 K 折交叉验证评估感知机模型的性能。

#### 函数定义

```python
def cross_validate(model_class, X, y, k=5):
```

- `model_class` (class): 要评估的模型类。
- `X` (list of list of floats): 输入特征数据。
- `y` (list of ints): 标签数据。
- `k` (int): 交叉验证的折数，默认为 5。

返回值:

- `float`: 模型的平均准确率。
- `float`: 准确率的标准差。

