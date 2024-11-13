"""
Created on 11 13, 2024
@author: <Cui>
@bref: 模型训练操作，推理，可视化等操作
"""

import torch
from net_result import NetResult
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

import matplotlib.pyplot as plt
import math
import os


class NetOperator:
    def __init__(self, net, train_data, lr=0.001, save_model=True, save_path="../data/model/model.pt"):
        self.net = net
        self.train_data = train_data
        self.lr = lr
        self.save_model = save_model
        self.save_path = save_path

    def train(self, epoch_num):
        """
        训练函数: 训练一个模型

        参数:
        - epoch_num: 训练次数
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        net_result = NetResult(self.net, self.train_data)

        for epoch in range(epoch_num):
            for (x, y) in self.train_data:
                self.net.zero_grad()
                output = self.net.forward(x.view(-1, 28 * 28))
                loss = torch.nn.functional.nll_loss(output, y)
                loss.backward()
                optimizer.step()

            print("epoch", epoch, "accuracy:", net_result.evaluate())

        if self.save_model is True:
            torch.save(self.net.state_dict(), self.save_path)

    def infer(self, image_path):
        """
        推理函数: 给定一个图片路径，输出预测的类别标签

        参数:
        - image_path: 推理图像路径
        """

        # 设置模型为评估模式
        self.net.eval()

        # 读取并预处理图片
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 转为灰度图（如果需要）
            transforms.Resize((28, 28)),  # 调整为28x28的大小
            transforms.ToTensor(),  # 转为Tensor
            transforms.Normalize(mean=[0.1307], std=[0.3081])  # 使用MNIST的归一化值
        ])

        # 打开图片
        image = Image.open(image_path)

        # 对图片进行转换
        image_tensor = transform(image)

        # 展平图像为1x784（batch_size, 784）张量
        image_tensor = image_tensor.view(1, -1)  # 添加batch维度，并将28x28图像展平为一维

        # 禁用梯度计算
        with torch.no_grad():
            # 进行推理
            output = self.net(image_tensor)  # 直接传入模型，保持 batch 维度
            print(output)

            # 使用 softmax 转换为概率分布
            output_probs = F.softmax(output, dim=1)
            print(output_probs)

            # 获取最大值索引，这就是预测类别
            predicted_class = output_probs.argmax(dim=1).item()

        return predicted_class

    def visualize_predictions(self, image_dir, grid_size=None):
        """
        可视化预测结果，并将图像排列成正方形网格

        参数:
        - image_dir: 图像目录
        - grid_size: 如果传入此参数，将会使用指定的网格尺寸
        """

        # 获取目录中的所有图像文件
        image_dir = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        image_path_predictions = []

        for image_path in image_dir:
            image_path_predictions.append((image_path, self.infer(image_path)))

        num_images = len(image_dir)

        if not grid_size:
            grid_size = self._get_grid_size(num_images)  # 自动计算网格尺寸

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
        axes = axes.flatten()  # 展平，以便于按顺序填充每个子图

        # 遍历每个图像
        for i, (image_path, pred) in enumerate(image_path_predictions):
            image = Image.open(image_path).convert("RGB")
            axes[i].imshow(image)
            axes[i].set_title(f'Pred: {pred}')
            axes[i].axis('off')  # 关闭坐标轴显示

        # 如果图像数量少于网格大小，隐藏多余的子图
        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def _get_grid_size(self, num_images):
        """
        计算合适的网格大小，尽量形成一个正方形
        """
        return int(math.ceil(math.sqrt(num_images)))