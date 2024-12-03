#pip install fasttext
import fasttext

# 训练模型
model = fasttext.train_supervised("train.txt", lr=0.1, epoch=10, wordNgrams=2)

# 测试模型性能
result = model.test("test.txt")
#print(f"Precision: {result.precision:.4f}")
#print(f"Recall: {result.recall:.4f}")
#print(f"Number of examples: {result.nexamples}")

# 分类预测
texts_to_predict = ["soccer game score", "new programming languages", "yoga and meditation"]
predictions = [model.predict(text) for text in texts_to_predict]

# 输出预测结果
for text, (label, prob) in zip(texts_to_predict, predictions):
    print(f"Text: {text} => Predicted Label: {label[0].replace('__label__', '')}, Confidence: {prob[0]:.4f}")
