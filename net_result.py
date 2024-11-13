"""
Created on 11 13, 2024
@author: <Cui>
@bref: 模型训练的结果，包括准确率的评估和结果的可视化
"""

import torch
import matplotlib.pyplot as plt


class NetResult:
    def __init__(self, net, data):
        '''

        :param net: 模型
        :param data: 数据可以是训练数据，也可以是测试数据
        '''
        self.net = net
        self.data = data

    def evaluate(self):
        n_correct = 0
        n_total = 0

        with torch.no_grad():
            for (x, y) in self.data:
                outputs = self.net.forward(x.view(-1, 28 * 28))
                for i, output in enumerate(outputs):
                    if torch.argmax(output) == y[i]:
                        n_correct += 1
                    n_total += 1

        return n_correct / n_total

    def evaluate_acc(self, model_path):
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()  # 切换到评估模式

        return self.evaluate()

    def show_result(self, number, show=True):

        for (n, (x, _)) in enumerate(self.data):
            if n > (number - 1):
                break

            predict = torch.argmax(self.net.forward(x[0].view(-1, 28 * 28)))
            plt.figure(n)
            plt.imshow(x[0].view(28, 28))
            plt.title("prediction: " + str(int(predict)))

        if show is True:
            plt.show()