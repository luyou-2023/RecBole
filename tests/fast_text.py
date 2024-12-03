#pip install fasttext
import random

# 模拟数据
categories = ["sports", "technology", "health"]
train_texts = [
    ("football basketball soccer", "sports"),
    ("artificial intelligence machine learning", "technology"),
    ("fitness diet exercise", "health"),
    ("baseball hockey tennis", "sports"),
    ("software programming coding", "technology"),
    ("nutrition wellness lifestyle", "health")
]
test_texts = [
    ("tennis match and cricket", "sports"),
    ("new ai algorithms", "technology"),
    ("healthy eating tips", "health")
]

# 保存训练数据到 train.txt 文件
with open("train.txt", "w") as train_file:
    for text, label in train_texts:
        train_file.write(f"__label__{label} {text}\n")

# 保存测试数据到 test.txt 文件
with open("test.txt", "w") as test_file:
    for text, label in test_texts:
        test_file.write(f"__label__{label} {text}\n")
