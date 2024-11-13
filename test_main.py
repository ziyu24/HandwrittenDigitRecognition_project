"""
Created on 11 13, 2024
@author: <Cui>
@bref: 测试全部模块
"""


from data_loader import DataLoaderMNIST
from net_constructor import LeNet
from net_result import NetResult
from net_operator import NetOperator
from confusion_matrix_visualizer import ConfusionMatrixVisualizer


def main():
    data_loader_mnist = DataLoaderMNIST("../data", 15)

    train_data = data_loader_mnist.get_data_loader(True)
    test_data = data_loader_mnist.get_data_loader(False)

    net = LeNet()

    net_result = NetResult(net, test_data)
    print("before train accuracy:", net_result.evaluate())

    net_operator = NetOperator(net, train_data)
    # net_operator.train(1)

    # 1.得到训练模型，模型通过 NetOperator 一般自动保存在 "../data/model/model.pt" 中

    # 2.评估模型精度
    print("after train accuracy:", net_result.evaluate_acc("../data/model/model.pt"))
    net_result.show_result(2)

    # 3.推理一张图片
    print("hand write 8 infer is: ", net_operator.infer("../data/test/test_3.png"))

    # 4.预测结果混淆矩阵可视化
    cm_visualizer = ConfusionMatrixVisualizer(net, test_data)
    cm_visualizer.visualize()

    # 5.数据的可视化
    net_operator.visualize_predictions("../data/test")


if __name__ == "__main__":
    main()
