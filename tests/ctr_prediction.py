import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 随机生成一些模拟数据
def generate_data(n_samples=10000):
    np.random.seed(42)

    # 用户特征
    user_age = np.random.randint(18, 65, size=n_samples)
    user_gender = np.random.randint(0, 2, size=n_samples)

    # 商品特征
    item_category = np.random.randint(0, 10, size=n_samples)  # 10个类别
    item_price = np.random.uniform(5, 200, size=n_samples)

    # 广告特征
    ad_position = np.random.randint(0, 5, size=n_samples)  # 5个广告位置

    # 目标变量：点击率（0或1）
    click = np.random.randint(0, 2, size=n_samples)

    # 特征矩阵
    features = np.column_stack([user_age, user_gender, item_category, item_price, ad_position])

    return features, click

# 数据预处理
def preprocess_data(features, labels):
    # 特征标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # 转换为 PyTorch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# 构建CTR模型
class CTRModel(nn.Module):
    def __init__(self, input_dim):
        super(CTRModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 第一层，输入特征维度到 64
        self.fc2 = nn.Linear(64, 32)         # 第二层，64 到 32
        self.fc3 = nn.Linear(32, 1)          # 输出层，输出 1 个值（点击概率）
        self.sigmoid = nn.Sigmoid()          # Sigmoid 激活函数，输出概率值

    def forward(self, x):
        x = torch.relu(self.fc1(x))          # 第一层 ReLU 激活
        x = torch.relu(self.fc2(x))          # 第二层 ReLU 激活
        x = self.fc3(x)                      # 输出层
        return self.sigmoid(x)               # 输出点击概率

# 设置超参数
learning_rate = 0.001
batch_size = 64
epochs = 10

# 生成数据
features, labels = generate_data(n_samples=10000)

# 数据预处理
X_train, X_test, y_train, y_test = preprocess_data(features, labels)

# 初始化模型
input_dim = X_train.shape[1]  # 输入特征的维度
model = CTRModel(input_dim)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        # 取出批次数据
        inputs = X_train[i:i + batch_size]
        labels = y_train[i:i + batch_size]

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(X_train):.4f}")

# 测试模型
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = outputs.round()  # 将概率值转为 0 或 1

    # 计算准确率
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
