import torch

# 示例：简单的自动微分
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()

# 打印梯度
print(x.grad)  # 输出应为 2*x + 3 在 x=2 时的值，即 7
