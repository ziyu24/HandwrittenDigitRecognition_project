"""
Created on 11 13, 2024
@author: <Cui>
@bref: 模型训练操作，推理，可视化等操作
"""

import torch
from torchvision.utils import save_image

from project.src.common.config import config_yaml
from project.src.val import evaluate_train


def train(model, train_data, test_data,
          epoch_num=config_yaml['train']['epochs'],
          lr=config_yaml['optimizer']['lr'],
          save_model=config_yaml['model']['save_model'],
          save_path=config_yaml['model_save_path']):
    """
    训练函数: 训练一个模型
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        for (x, y) in train_data:
            model.zero_grad()
            output = model.forward(x.view(-1, config_yaml['net']['input_size']))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()

        print("epoch", epoch, "accuracy:", evaluate_train(model, test_data))

    if save_model is True:
        torch.save(model.state_dict(), save_path)
