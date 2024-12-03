import torch

# 数据
x_data = torch.tensor([1.0, 2.0, 3.0])
y_data = torch.tensor([2.0, 4.0, 6.0])

# 初始化权重
weight = torch.tensor([1.0], requires_grad=True)

# 学习率
learning_rate = 0.01

# 前向传播
def forward(x):
    return x * weight

# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return ((y_pred - y) ** 2).mean()

# 训练循环
for epoch in range(100):  # 训练 10 次
    # 1. 计算损失
    l = loss(x_data, y_data)

    # 2. 反向传播，计算梯度
    l.backward()

    # 3. 手动更新权重
    with torch.no_grad():  # 禁止梯度跟踪
        weight -= learning_rate * weight.grad  # 更新权重

    # 4. 清除梯度，否则梯度会累积
    weight.grad.zero_()

    print(f"Epoch {epoch+1}: Loss = {l.item()}, Weight = {weight.item()}")
