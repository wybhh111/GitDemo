import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtractor:
    def __init__(self, method='count'):
        """
        初始化特征提取器，可以选择 'count' 或 'tf-idf' 方法
        """
        self.method = method
        if self.method == 'tf-idf':
            self.vectorizer = TfidfVectorizer(tokenizer=self._jieba_tokenizer, token_pattern=None)
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(tokenizer=self._jieba_tokenizer, token_pattern=None)
        else:
            raise ValueError("无效的方法。请选择 'tf-idf' 或 'count'。")

    def _jieba_tokenizer(self, text):
        """使用jieba进行分词，并过滤掉短词"""
        text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)  # 适当的正则表达式清理
        words = cut(text)
        return [word for word in words if len(word) > 1]

    def fit_transform(self, documents):
        """适合并转换文档"""
        return self.vectorizer.fit_transform(documents)

    def get_feature_names(self):
        """获取特征名称"""
        return self.vectorizer.get_feature_names_out()


def get_documents(filepath_list):
    """读取多个文件并返回文档列表"""
    documents = []
    for filename in filepath_list:
        with open(filename, 'r', encoding='utf-8') as fr:
            content = fr.read()
            documents.append(content)
    return documents


# 构建邮件文件名
filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
documents = get_documents(filename_list)

# 参数选择特征提取方式
feature_extraction_method = 'tf-idf'  # 可选 'tf-idf' 或 'count'
extractor = FeatureExtractor(method=feature_extraction_method)

# 提取特征
features = extractor.fit_transform(documents)

# 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
labels = np.array([1] * 127 + [0] * 24)

# 训练模型
model = MultinomialNB()
model.fit(features, labels)


def predict(filename):
    """对未知邮件分类"""
    # 读取并处理未知邮件
    with open(filename, 'r', encoding='utf-8') as fr:
        content = fr.read()
    current_vector = extractor.vectorizer.transform([content])

    # 预测结果
    result = model.predict(current_vector)
    return '垃圾邮件' if result == 1 else '普通邮件'


# 测试分类
test_files = ['邮件_files/151.txt', '邮件_files/152.txt', '邮件_files/153.txt', '邮件_files/154.txt',
              '邮件_files/155.txt']
for test_file in test_files:
    print(f'{test_file} 分类情况: {predict(test_file)}')  