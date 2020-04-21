import torchvision.transforms as transforms
import torch
import os
from datasets import FaceDataset
import torch.nn as nn
import torch.optim as optim
from model.AE import AutoEncoder
from tools import weights_init, show_data, plot_loss


"""超参数设置"""
data_root = os.path.abspath(os.path.dirname(__file__)) + '/data/face/'
image_size = (96, 96)           # 输入图片尺寸
batch_size = 600                # 训练时的batch大小
workers = 2                     # dataloader的workers数量
in_channels = 3                 # autoencoder输入通道数
out_channels = 100              # autoencoder输出通道数
lr = 0.0002                     # 学习率
beta1 = 0.5                     # adam优化器的参数
epochs = 50                     # 训练轮数
device = torch.device('cpu')
gpu_num = 0
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_num = torch.cuda.device_count()
# 损失函数
criterion = nn.MSELoss()


"""数据集构造"""
transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

faceDataset = FaceDataset(root=data_root, transforms=transforms)
print('数据集容量：', len(faceDataset))
dataloader = torch.utils.data.DataLoader(
    faceDataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers)

losses = []


def train(autoencoder, optimizer, start_epoch):
    for epoch in range(start_epoch, epochs):
        for i, data in enumerate(dataloader):
            autoencoder.zero_grad()
            img = data.to(device)
            _, fake_img = autoencoder(img)
            loss = criterion(img, fake_img)
            loss.backward()
            optimizer.step()

            losses.append(loss)

            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                      % (epoch, epochs, i, len(dataloader),
                         loss.item()))

        test_noise = torch.randn(64, out_channels, 1, 1, device=device)
        with torch.no_grad():
            if isinstance(autoencoder, torch.nn.DataParallel):
                fake = autoencoder.module.decoder(test_noise).detach().cpu()
        show_data(fake, 'epoch %d: generated images' % epoch)

        torch.save({
            'epoch': epoch,
            'loss': losses,
            'autoencoder_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, './checkpoints/AE/epoch_%d.pth' % epoch)


if __name__ == "__main__":

    batch = next(iter(dataloader))
    show_data(batch[:64], description='training images')

    autoencoder = AutoEncoder(in_channels, out_channels).to(device)
    autoencoder.apply(weights_init)
    print(autoencoder)

    # 是否使用多块GPU
    if (device.type == 'cuda') and (gpu_num > 1):
        print("使用%d块GPU训练!" % gpu_num)
        autoencoder = nn.DataParallel(autoencoder)

    optimizer = optim.Adam(autoencoder.parameters(), lr=lr, betas=(beta1, 0.999))

    start_epoch = 0

    checkpoints = torch.load('./checkpoints/AE/epoch_49.pth')
    autoencoder.load_state_dict(checkpoints['autoencoder_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    start_epoch = checkpoints['epoch']
    losses = checkpoints['loss']

    # train(autoencoder, optimizer, start_epoch)

    loss = {'loss_list': losses, 'description': 'ae_loss'}
    plot_loss(loss)

    test_noise = torch.randn(64, out_channels, 1, 1, device=device)
    with torch.no_grad():
        if isinstance(autoencoder, torch.nn.DataParallel):
            fake = autoencoder.module.decoder(test_noise).detach().cpu()
    show_data(fake, 'final generated images')















