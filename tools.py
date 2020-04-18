import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


# 初始化网络卷积层和归一化层的参数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 展示训练数据
def show_data(data, description):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(description)
    plt.imshow(np.transpose(vutils.make_grid(data,
                                             padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


# 打印loss曲线
def plot_loss(*params):
    plt.figure()
    for loss in params:
        description = loss['description']
        loss_list = loss['loss_list']
        plt.plot(list(range(len(loss_list))), loss_list, label=description)
    plt.show()





