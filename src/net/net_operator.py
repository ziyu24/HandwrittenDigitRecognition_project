"""
Created on 11 13, 2024
@author: <Cui>
@bref: 模型训练操作，推理，可视化等操作
"""

import torch
from torchvision.utils import save_image

from project.src.net.net_result import NetResult
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

import matplotlib.pyplot as plt
import math
import os

import seaborn as sns
from sklearn.metrics import confusion_matrix

from project.src.common.config import config_yaml


def _get_grid_size(num_images):
    """
    计算合适的网格大小，尽量形成一个正方形
    """
    return int(math.ceil(math.sqrt(num_images)))


class NetOperator:
    def __init__(self, net, train_data, test_data,
                 classes=config_yaml['data']['class_num'],
                 lr=config_yaml['optimizer']['lr'],
                 save_model=config_yaml['model']['save_model'],
                 save_path=config_yaml['model_save_path']):
        self.net = net
        self.train_data = train_data
        self.test_data = test_data
        self.classes = [str(i) for i in range(classes)]  # 类别标签，例如 '0', '1', ..., '9'
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
                output = self.net.forward(x.view(-1, config_yaml['net']['input_size']))
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
            transforms.Resize((config_yaml['net']['height'], config_yaml['net']['width'])),  # 调整为28x28的大小
            transforms.ToTensor(),  # 转为Tensor
            # transforms.Normalize(mean=[config_yaml['train']['mean']], std=[config_yaml['train']['std']])  # 使用MNIST的归一化值
        ])

        # 打开图片
        image = Image.open(image_path)
        # 对图片进行转换
        image_tensor = transform(image)
        # # 保存图像的路径
        # image_path = config_yaml['test_data_dir'] + '/../out_put_1.jpg'
        # save_image(image_tensor, image_path)
        # 展平图像为1x784（batch_size, 784）张量
        image_tensor = image_tensor.view(1, -1)  # 添加batch维度，并将28x28图像展平为一维

        # 禁用梯度计算
        with torch.no_grad():
            # 进行推理
            output = self.net(image_tensor)  # 直接传入模型，保持 batch 维度
            # print(output)

            # 使用 softmax 转换为概率分布
            output_probs = F.softmax(output, dim=1)
            print(output_probs)

            # 获取最大值索引，这就是预测类别
            predicted_class = output_probs.argmax(dim=1).item()

        return predicted_class

    def _get_grid_size(num_images):
        """根据图像数量自动计算合适的网格大小"""
        grid_size = int(num_images ** 0.5)  # 取平方根作为网格大小的初步估算
        if grid_size * grid_size < num_images:
            grid_size += 1  # 如果网格的总大小不足以放下所有图像，增加网格大小
        return grid_size

    def visualize_predictions(self, image_dir, grid_size=None):
        """
        可视化预测结果，并将图像排列成正方形网格

        参数:
        - image_dir: 图像目录
        - grid_size: 如果传入此参数，将会使用指定的网格尺寸
        """
        # 获取目录中的所有图像文件
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                       f.endswith(('.png', '.jpg', '.jpeg'))]

        image_path_predictions = []

        for image_path in image_files:
            # 获取每个图像的预测结果
            image_path_predictions.append((image_path, self.infer(image_path)))

        num_images = len(image_path_predictions)

        if num_images == 1:
            # 只有一张图像时，不使用子图，直接绘制
            image_path, pred = image_path_predictions[0]
            image = Image.open(image_path).convert("RGB")
            plt.imshow(image)
            plt.title(f'Pred: {pred}')
            plt.axis('off')
            plt.show()
            return

        # 如果没有提供grid_size，自动计算网格大小
        if not grid_size:
            grid_size = _get_grid_size(num_images)

        # 创建子图
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
        axes = axes.flatten()  # 将二维的axes展平，以便按顺序填充每个子图

        # 遍历所有图像并绘制
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

    def visualize_confusion_matrix(self):
        """
        公共方法：可视化混淆矩阵
        """
        y_true, y_pred = self._get_predictions()  # 获取真实标签和预测标签
        self._plot_confusion_matrix(y_true, y_pred)  # 绘制混淆矩阵

    def _get_predictions(self):
        """
        私有方法：获取所有预测结果和真实标签

        :return: 返回预测标签和真实标签
        """
        self.net.eval()  # 切换到评估模式
        all_labels = []
        all_preds = []

        with torch.no_grad():  # 禁用梯度计算
            for (x, y) in self.test_data:
                output = self.net(x.view(-1, config_yaml['net']['input_size']))  # 输入数据经过模型
                _, predicted = torch.max(output, 1)  # 获取最大概率的标签

                all_labels.extend(y.numpy())  # 真实标签
                all_preds.extend(predicted.numpy())  # 预测标签

        return all_labels, all_preds

    def _plot_confusion_matrix(self, y_true, y_pred):
        """
        私有方法：绘制混淆矩阵的热力图

        :param y_true: 真实标签
        :param y_pred: 预测标签
        """
        cm = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
