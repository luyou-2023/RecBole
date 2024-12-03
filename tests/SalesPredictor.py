import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# 数据生成
def generate_data(num_samples=1000):
    """
    构建模拟数据，特征 x 包含 3 个维度，与目标 y 存在特定线性或非线性关系
    """
    x = np.random.rand(num_samples, 3) * 10  # 3 个特征，每个值在 [0, 10] 范围内
    # 构造目标值 y = 3*x1 + 2*x2^2 - 5*sin(x3) + 噪声
    y = 3 * x[:, 0] + 2 * x[:, 1]**2 - 5 * np.sin(x[:, 2]) + np.random.randn(num_samples) * 2
    return x, y

# 生成训练和测试数据
train_x, train_y = generate_data(800)  # 800 条训练数据
test_x, test_y = generate_data(200)    # 200 条测试数据

# 转换为 PyTorch 的张量
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)  # 添加一个维度
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1)

# 数据检查
print(f"训练数据形状: {train_x.shape}, {train_y.shape}")
print(f"测试数据形状: {test_x.shape}, {test_y.shape}")

# 构建数据加载器
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义网络模型
class SalesPredictor(nn.Module):
    def __init__(self):
        super(SalesPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64),  # 输入维度是 3，隐藏层 64 个节点
            nn.ReLU(),
            nn.Linear(64, 32),  # 第二个隐藏层
            nn.ReLU(),
            nn.Linear(32, 1)    # 输出 1 个预测值
        )

    def forward(self, x):
        return self.network(x)

# 实例化模型
model = SalesPredictor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 50
loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 绘制损失曲线
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# 测试模型
model.eval()
with torch.no_grad():
    test_x, test_y = test_x.to(device), test_y.to(device)
    predictions = model(test_x)
    test_loss = criterion(predictions, test_y).item()

print(f"测试集上的均方误差: {test_loss:.4f}")

# 可视化预测结果
predictions = predictions.cpu().numpy()
test_y = test_y.cpu().numpy()

plt.scatter(range(len(test_y)), test_y, label="True Values", color="blue", alpha=0.6)
plt.scatter(range(len(test_y)), predictions, label="Predictions", color="red", alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.title('True vs Predicted Sales')
plt.legend()
plt.show()
