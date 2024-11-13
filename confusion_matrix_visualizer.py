"""
Created on 11 13, 2024
@author: <Cui>
@bref: 混淆矩阵可视化类的实现，以及测试
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_loader import DataLoaderMNIST
from net_constructor import LeNet


class ConfusionMatrixVisualizer:
    def __init__(self, model, test_data, num_classes=10):
        """
        初始化混淆矩阵可视化器

        :param model: 训练好的 PyTorch 模型
        :param test_data: 测试数据集
        :param num_classes: 类别数量，默认是 10（适用于 MNIST）
        """
        self.model = model
        self.test_data = test_data
        self.num_classes = num_classes
        self.classes = [str(i) for i in range(self.num_classes)]  # 类别标签，例如 '0', '1', ..., '9'

    def visualize(self):
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
        self.model.eval()  # 切换到评估模式
        all_labels = []
        all_preds = []

        with torch.no_grad():  # 禁用梯度计算
            for (x, y) in self.test_data:
                output = self.model(x.view(-1, 28 * 28))  # 输入数据经过模型
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


if __name__ == '__main__':

    # 加载完整模型
    net = LeNet()
    net.load_state_dict(torch.load('../data/model/model_complete.pth'))
    net.eval()  # 切换到评估模式

    data_loader = DataLoaderMNIST("../data", 15)
    test_data = data_loader.get_data_loader(False)

    # 假设 net 是你的训练好的模型，test_data 是你的测试数据集
    cm_visualizer = ConfusionMatrixVisualizer(model=net, test_data=test_data, num_classes=10)
    cm_visualizer.visualize()  # 调用公共方法进行可视化
