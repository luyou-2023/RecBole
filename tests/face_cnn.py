import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
import cv2
import numpy as np
import os

# 超参数
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# 数据目录，需包含 'train' 和 'test' 子目录
data_dir = '/Users/luyou/code_work/shopastro/rust_code/RecBole/tests/data_faces'

# 数据预处理（包括调整大小、转换为张量）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整到输入大小
    transforms.ToTensor()          # 转为 PyTorch 张量
])

# 加载数据集，ImageFolder 会自动解析文件夹名为标签
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

# 数据加载器，分批次加载数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义自定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 输入通道数3，输出通道数16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)      # 最大池化层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 第二层卷积
        self.fc1 = nn.Linear(32 * 56 * 56, 128)                           # 全连接层 1
        self.fc2 = nn.Linear(128, num_classes)                           # 全连接层 2 输出类别数

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 第一层卷积 + ReLU + 池化
        x = self.pool(torch.relu(self.conv2(x)))  # 第二层卷积 + ReLU + 池化
        x = x.view(-1, 32 * 56 * 56)              # 展平成一维向量
        x = torch.relu(self.fc1(x))              # 全连接层 1
        x = self.fc2(x)                          # 全连接层 2 输出
        return x


# 使用自定义模型
model = SimpleCNN(num_classes=len(train_dataset.classes))

# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'face_recognition_model.pth')
print("模型已保存为 face_recognition_model.pth")


# 实时人脸检测与识别
def face_recognition_live():
    # 加载 Haar 特征级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 加载训练好的模型
    model.load_state_dict(torch.load('face_recognition_model.pth'))
    model.eval()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # 绘制矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 识别每张脸
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))
            face = transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(face)
                _, predicted = torch.max(outputs, 1)
                label = train_dataset.classes[predicted]
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def face_recognition_image(image_path):
    # 加载图片
    face = cv2.imread(image_path)  # 假设你使用 OpenCV 加载的图片
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

    # 确保图片类型为 PIL.Image
    if isinstance(face, np.ndarray):
        face = Image.fromarray(face)

    # 定义预处理变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小到 224x224
        transforms.ToTensor(),         # 转换为 PyTorch 张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 应用变换
    face = transform(face).unsqueeze(0).to(device)

    # 使用自定义模型结构
    model = SimpleCNN(num_classes=len(train_dataset.classes))  # 使用训练时的模型结构
    model.load_state_dict(torch.load("face_recognition_model.pth"))  # 加载权重
    model.eval()

    # 使用模型进行预测
    with torch.no_grad():
        output = model(face)
        _, predicted = torch.max(output, 1)

    # return predicted.item()
    return train_dataset.classes[predicted.item()]


# 启动实时人脸识别
lable = face_recognition_image("/Users/luyou/code_work/shopastro/rust_code/RecBole/tests/data_faces/train/Albert_Costa/Albert_Costa_0001.jpg")
print(lable)
