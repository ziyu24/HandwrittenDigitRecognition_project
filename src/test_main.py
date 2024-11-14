"""
Created on 11 13, 2024
@author: <Cui>
@bref: 测试全部模块
"""
import os

from project.src.common.data_loader import DataLoaderMNIST
from project.src.net.net_constructor import LeNet
from project.src.net.net_result import NetResult
from project.src.net.net_operator import NetOperator
from project.src.common.config import config_yaml


def main():
    net = LeNet()

    data_loader_mnist = DataLoaderMNIST(config_yaml['dataset_dir'], config_yaml['train']['batch_size'])
    test_data = data_loader_mnist.get_data_loader(False)

    net_result = NetResult(net, test_data)
    print("before train accuracy:", net_result.evaluate())

    train_data = data_loader_mnist.get_data_loader(True)
    net_operator = NetOperator(net, train_data, test_data)
    # net_operator.train(config_yaml['train']['epochs'])

    # 1.得到训练模型，模型通过 NetOperator 一般自动保存在 "../data/model/model.pt" 中

    # 2.评估模型精度
    print("after train accuracy:", net_result.evaluate_acc(config_yaml['model_save_path']))
    net_result.show_result(2)

    # 3.推理一张图片
    print("hand write 3 infer is: ", net_operator.infer(config_yaml['test_data_dir'] + '/test_1.png'))

    # 4.预测结果混淆矩阵可视化
    net_operator.visualize_confusion_matrix()

    # 5.数据的可视化
    net_operator.visualize_predictions(config_yaml['test_data_dir'])


if __name__ == "__main__":
    main()
