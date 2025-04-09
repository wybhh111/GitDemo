import re
import numpy as np
from jieba import cut
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report  # 导入classification_report

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
        text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)
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

# 通过SMOTE平衡数据
smote = SMOTE()
features_resampled, labels_resampled = smote.fit_resample(features, labels)

# 训练模型
model = MultinomialNB()
model.fit(features_resampled, labels_resampled)

def predict(filename):
    """对未知邮件分类"""
    with open(filename, 'r', encoding='utf-8') as fr:
        content = fr.read()
    current_vector = extractor.vectorizer.transform([content])
    result = model.predict(current_vector)
    return 1 if result == 1 else 0  # 返回整数标签，1表示垃圾邮件，0表示普通邮件

# 测试分类
test_files = ['邮件_files/151.txt', '邮件_files/152.txt', '邮件_files/153.txt', '邮件_files/154.txt',
              '邮件_files/155.txt']

# 收集预测结果和实际标签
predictions = []
true_labels = []

# 假设文件名中包含“垃圾邮件”标记来判断真实标签
for test_file in test_files:
    true_label = 1 if '垃圾邮件' in test_file else 0
    true_labels.append(true_label)
    prediction = predict(test_file)
    predictions.append(prediction)

# 输出分类评估报告
print(classification_report(true_labels, predictions, target_names=['普通邮件', '垃圾邮件']))