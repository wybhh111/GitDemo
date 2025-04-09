# 代码核心功能说明
## 1. 导入库  

```python  
import pandas as pd  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```
### pandas用于数据处理和分析，CountVectorizer 和 TfidfVectorizer这两个类用于文本特征提取。
## 2. 定义 FeatureExtractor 类
```python
class FeatureExtractor:  
    def __init__(self, method='tf-idf'):  
        # 选择特征提取方法  
        self.method = method  
        if self.method == 'tf-idf':  
            self.vectorizer = TfidfVectorizer()  
        elif self.method == 'count':  
            self.vectorizer = CountVectorizer()  
        else:  
            raise ValueError("Invalid method. Choose 'tf-idf' or 'count'.")
  ```
### 构造函数，根据 method 创建相应的特征提取器，如果 method 无效，抛出 ValueError 异常。
## 3. 特征提取方法
```python
def fit_transform(self, documents):  
    # 适合并转换文档  
    return self.vectorizer.fit_transform(documents)
  ```
### fit_transform 方法，使用所选的 vectorizer 对文档进行拟合和转换，返回特征矩阵。
## 4. 获取特征名称
```python
def get_feature_names(self):  
    # 获取特征名称  
    return self.vectorizer.get_feature_names_out()
```
###  get_feature_names 方法，返回当前特征提取器所提取出的特征名称。
## 5. 示例数据
```python
# 示例数据  
documents = [  
    "这是第一个文档。",  
    "这是第二个文档，这是一个示例。",  
    "这是第三个文档。",  
]
```
### documents，包含三个中文字符串的列表，作为输入数据进行特征提取。
## 6. 使用特征提取器
```python
# 选择特征提取方式  
method = 'tf-idf'  # 可选 'tf-idf' 或 'count'；根据需要修改  

# 创建特征提取器实例  
extractor = FeatureExtractor(method)  

# 提取特征  
features = extractor.fit_transform(documents)
```
### 设置 method 为 'tf-idf' 或 'count'，创建 FeatureExtractor 的实例 extractor，调用 fit_transform 方法，使用示例数据提取特征。
## 7. 输出特征名称和特征矩阵
```python
# 输出特征名称和特征矩阵  
print("特征名称:", extractor.get_feature_names())  
print("特征矩阵:\n", features.toarray())
```
### 调用 get_feature_names() 获取特征名称并输出，使用 toarray() 方法将特征矩阵转换为稠密数组并输出。
![1](https://github.com/wybhh111/GitDemo/blob/master/images/2025-04-09%20224945.png)
![2](https://github.com/wybhh111/GitDemo/blob/master/images/2025-04-09%20230017.png)
# 高频词/TF-IDF两种特征模式的切换方法
### 特征选择，监测模型在训练集和验证集上的表现，选择适合的特征方法。如果需要更强调文本的主题和语义信息，可以选择TF-IDF。混合使用，在某些情况下可以同时提取高频词和TF-IDF特征，将两者组合成训练集的特征。例如，可以将高频词作为基础特征，TF-IDF作为权重调整的特征，从而增强模型的表现。模型调优，在项目中进行实验，通过交叉验证评估不同特征提取方法的效果，选择最佳方案。动态切换，构建一个动态特征提取管道，根据输入数据的特点自动切换使用高频词还是TF-IDF。
# 样本平衡处理
![3](https://github.com/user-attachments/assets/6fc7d197-68ba-41d6-a505-cae0505b7a24)
# 增加模型评估指标
![4](https://github.com/user-attachments/assets/c5ea42fa-daee-497f-8054-615283ecd3e2)
