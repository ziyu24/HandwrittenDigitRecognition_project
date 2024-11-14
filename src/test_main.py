"""
Created on 11 13, 2024
@author: <Cui>
@bref: 测试全部模块
"""

from project.src.datasets.data_loader import DataLoaderMNIST
from project.src.models.net_constructor import LeNet
from project.src.common.config import config_yaml
from project.src.tool.visualize import show_test_result, visualize_confusion_matrix, visualize_predictions

from val import evaluate_train, evaluate_acc
from train import train
from infer import infer


def main():
    model = LeNet()

    data_loader_mnist = DataLoaderMNIST(config_yaml['dataset_dir'], config_yaml['train']['batch_size'])
    test_data = data_loader_mnist.get_data_loader(False)

    print("before train accuracy:", evaluate_train(model, test_data))

    train_data = data_loader_mnist.get_data_loader(True)
    train(model, train_data, test_data)

    # 1.得到训练模型，模型通过 NetOperator 一般自动保存在 "../data/model/model.pt" 中

    # 2.评估模型精度
    print("after train accuracy:", evaluate_acc(model, test_data))
    show_test_result(model, test_data, 2)

    # 3.推理一张图片
    print("hand write 3 infer is: ", infer(model, config_yaml['test_data_dir'] + '/test_1.png'))

    # 4.预测结果混淆矩阵可视化
    visualize_confusion_matrix(model, test_data, config_yaml['data']['class_num'])

    # 5.数据的可视化
    visualize_predictions(model, config_yaml['test_data_dir'])


if __name__ == "__main__":
    main()
