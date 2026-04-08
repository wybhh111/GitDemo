import pandas as pd
from gensim.models import Word2Vec

# ==========================================
# 第一步：读取并处理《弗兰肯斯坦》数据
# ==========================================
print("正在读取弗兰肯斯坦数据...")

corpus = []

try:
    # 读取 CSV (注意：这里使用你上传的文件名)
    df = pd.read_csv('frankenstein_with_splits.csv')

    # 遍历每一行数据
    for _, row in df.iterrows():
        # 获取上下文和目标词
        context_str = str(row['context']).strip()
        target_str = str(row['target']).strip()

        # 跳过空值
        if context_str == 'nan' or target_str == 'nan':
            continue

        # 用空格分割单词 (英文数据)
        context_words = context_str.split() if context_str else []
        target_words = target_str.split() if target_str else []

        # 合并成一个句子: 上下文词 + 目标词
        sentence = context_words + target_words

        if len(sentence) > 0:
            corpus.append(sentence)

    print(f"数据读取完成，共构建 {len(corpus)} 个训练样本（句子）。")

except Exception as e:
    print(f"读取数据出错: {e}")
    print("请确保 'frankenstein_with_splits.csv' 在当前目录下。")

# ==========================================
# 第二步：训练 Skip-Gram 模型
# ==========================================
print("\n正在训练 Skip-Gram 模型...")

model = Word2Vec(
    sentences=corpus,
    sg=1,             # Skip-Gram
    vector_size=300,  # 词向量维度
    window=5,         # 窗口大小
    min_count=1,      # 忽略低频词 (设为1确保所有词都被保留)
    workers=4,        # 线程数
    epochs=10         # 迭代次数
)

print("1) Skip-Gram 模型训练完成。")

# ==========================================
# 第三步：执行具体任务
# ==========================================

# --- 任务 2: 输出“frankenstein”的词向量 ---
print("\n" + "=" * 30)
print("任务 2: 输出 'frankenstein' 的词向量")
target_word = 'frankenstein'
if target_word in model.wv:
    vector = model.wv[target_word]
    print(f"词向量形状: {vector.shape}")
    print(f"词向量数值 (前10维): {vector[:10]}")
else:
    print(f"错误：词汇 '{target_word}' 不在词汇表中。")

# --- 任务 3: 输出与“modern”最接近的3个词 ---
print("\n" + "=" * 30)
print("任务 3: 输出与 'modern' 语义最接近的3个词")
similar_word = 'modern'
if similar_word in model.wv:
    similar_words = model.wv.most_similar(similar_word, topn=3)
    for word, score in similar_words:
        print(f" - {word}: {score:.4f}")
else:
    print(f"错误：词汇 '{similar_word}' 不在词汇表中。")

# --- 任务 4: 计算相似度 ---
print("\n" + "=" * 30)
print("任务 4: 计算相似度")

# 根据你的数据内容，选择文中出现的词汇进行测试
word_pairs = [
    ('mary', 'shelley'),   # 人名关系
    ('petersburgh', 'england'), # 地理关系
    ('frankenstein', 'monster') # 书中概念关系 (如果数据中出现过)
]

for w1, w2 in word_pairs:
    if w1 in model.wv and w2 in model.wv:
        sim = model.wv.similarity(w1, w2)
        print(f"'{w1}' 和 '{w2}' 的相似度: {sim:.4f}")
    else:
        print(f"无法计算 '{w1}' 和 '{w2}' 的相似度（词汇缺失）。")

# --- 任务 5: 向量运算 ---
print("\n" + "=" * 30)
print("任务 5: 执行向量运算（示例：petersburgh + england - london = ?）")
try:
    # 这是一个类比：[Petersburgh 对于 England] 就像 [? 对于 England] (或者寻找地理关联)
    # 也可以理解为：找出与 'petersburgh' 和 'england' 关系密切，但排除 'london' 的词
    result = model.wv.most_similar(
        positive=['petersburgh', 'england'],
        negative=['london'],
        topn=1
    )
    best_word = result[0][0]
    best_score = result[0][1]
    print(f"运算结果：{best_word} (相似度: {best_score:.4f})")
except KeyError as e:
    print(f"运算失败：词汇 {e} 不在词汇表中。")