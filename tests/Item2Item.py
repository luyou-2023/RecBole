import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random

# 假设我们有一个包含用户和商品互动的行为数据集
# 示例行为数据 (user_id, item_id, interaction_value)
# interaction_value: 1代表点击，0代表没有点击
interactions = [
    (1, 101, 1), (1, 102, 1), (1, 103, 0),
    (2, 101, 0), (2, 102, 1), (2, 104, 1),
    (3, 101, 1), (3, 103, 1), (3, 104, 0),
    # 假设有更多的交互数据
]

# 预处理：将用户ID和商品ID映射到数字
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

user_ids = [x[0] for x in interactions]
item_ids = [x[1] for x in interactions]

user_encoder.fit(user_ids)
item_encoder.fit(item_ids)

user_ids_encoded = user_encoder.transform(user_ids)
item_ids_encoded = item_encoder.transform(item_ids)

# 转换为训练数据
train_data = [(user_ids_encoded[i], item_ids_encoded[i], interactions[i][2]) for i in range(len(interactions))]

# 定义数据集类
class Item2ItemDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, item_id, interaction = self.data[idx]
        return torch.tensor(user_id, dtype=torch.long), torch.tensor(item_id, dtype=torch.long), torch.tensor(interaction, dtype=torch.float32)

# 创建数据加载器
train_dataset = Item2ItemDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 定义一个简单的神经网络模型，用于学习商品之间的相似度
class Item2ItemModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=8):
        super(Item2ItemModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        interaction = torch.sigmoid(torch.sum(user_embedded * item_embedded, dim=1))  # 点积相似度
        return interaction

# 创建模型
num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
model = Item2ItemModel(num_users, num_items, embedding_dim=8)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for user, item, interaction in train_loader:
        optimizer.zero_grad()

        # 前向传播
        predicted_interaction = model(user, item)

        # 计算损失
        loss = criterion(predicted_interaction, interaction)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 测试模型 - 根据用户ID为其推荐商品
def recommend_items(user_id, top_k=3):
    model.eval()
    user_encoded = torch.tensor(user_encoder.transform([user_id]), dtype=torch.long)

    # 获取所有商品的预测互动分数
    item_ids = torch.arange(num_items)
    predictions = model(user_encoded.repeat(num_items), item_ids)

    # 获取前K个推荐商品
    recommended_item_ids = predictions.topk(top_k).indices
    recommended_item_ids = item_encoder.inverse_transform(recommended_item_ids.cpu().numpy())

    return recommended_item_ids

# 示例：为用户1推荐商品
recommended_items = recommend_items(1, top_k=3)
print("Recommended items for user 1:", recommended_items)
