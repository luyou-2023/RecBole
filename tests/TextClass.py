import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 模拟数据生成
def generate_text_data(num_samples=1000):
    """
    生成模拟的文本分类数据，文本与类别之间有一定的语义关联
    """
    classes = ["sports", "politics", "technology", "health"]
    data = []
    for _ in range(num_samples):
        label = np.random.choice(classes)
        if label == "sports":
            text = " ".join(np.random.choice(["football", "basketball", "team", "game", "score"], 5))
        elif label == "politics":
            text = " ".join(np.random.choice(["government", "election", "party", "policy", "leader"], 5))
        elif label == "technology":
            text = " ".join(np.random.choice(["computer", "AI", "software", "hardware", "network"], 5))
        elif label == "health":
            text = " ".join(np.random.choice(["doctor", "medicine", "health", "hospital", "care"], 5))
        data.append((text, label))
    return data

# 生成数据并划分训练和测试集
data = generate_text_data(1000)
texts, labels = zip(*data)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 标签编码
label_to_index = {label: idx for idx, label in enumerate(set(train_labels))}
index_to_label = {idx: label for label, idx in label_to_index.items()}
train_labels = [label_to_index[label] for label in train_labels]
test_labels = [label_to_index[label] for label in test_labels]

# 数据预处理：构造词汇表
def build_vocab(texts):
    vocab = set()
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        vocab.update(words)
    return {word: idx + 1 for idx, word in enumerate(vocab)}  # 0 保留给 PAD

vocab = build_vocab(train_texts)
vocab_size = len(vocab) + 1  # 包含 PAD 的索引

# 文本转为索引
def text_to_indices(text, vocab, max_len=10):
    indices = [vocab.get(word, 0) for word in re.findall(r'\b\w+\b', text.lower())]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [0] * (max_len - len(indices))
    return indices

train_indices = [text_to_indices(text, vocab) for text in train_texts]
test_indices = [text_to_indices(text, vocab) for text in test_texts]

# 构建数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = TextDataset(train_indices, train_labels)
test_dataset = TextDataset(test_indices, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 模型定义
class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(FastTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 嵌入层并计算平均池化
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        pooled = torch.mean(embeds, dim=1)  # 平均池化 [batch_size, embed_dim]
        return self.fc(pooled)

# 初始化模型
embed_dim = 50
num_classes = len(label_to_index)
model = FastTextClassifier(vocab_size, embed_dim, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        # 前向传播
        outputs = model(texts)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# 测试模型
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 打印分类报告
print(classification_report(all_labels, all_preds, target_names=index_to_label.values()))
