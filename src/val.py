"""
Created on 11 13, 2024
@author: <Cui>
@bref: 模型训练的结果，包括准确率的评估和结果的可视化
"""

import torch
import matplotlib.pyplot as plt
from project.src.common.config import config_yaml


def evaluate_train(model, test_data):
    """
    评估函数: 实时评估模型训练的精确度

    参数:
    - model: 待评估的模型
    - test_data: 评估的数据集
    """

    n_correct = 0
    n_total = 0

    model.eval()  # 切换到评估模式

    with torch.no_grad():
        for (x, y) in test_data:
            outputs = model.forward(x.view(-1, config_yaml['net']['input_size']))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1

    model.train()

    return n_correct / n_total


def evaluate_acc(model, test_data, use_save_model=config_yaml['evl']['use_save_model']):
    """
    评估函数: 评估训练后的模型精确度

    参数:
    - model: 待评估的模型
    - test_data: 评估的数据集
    """

    if use_save_model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(config_yaml['model_save_path'], map_location=device))
    model.eval()  # 切换到评估模式

    return evaluate_train(model, test_data)

