import torch

# 创建 Tensor
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# 构建一个简单的线性函数
y = w * x + b

# 计算损失
loss = y - 5

# 反向传播
loss.backward()

# 检查梯度
print(x.grad)  # dy/dx
print(w.grad)  # dy/dw
print(b.grad)  # dy/db
