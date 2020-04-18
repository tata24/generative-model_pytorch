import torchvision.transforms as transforms
import torch
import os
from datasets import FaceDataset
import torch.nn as nn
import torch.optim as optim
from model.DCGAN import Discriminator, Generator
from tools import weights_init, show_data, plot_loss


"""超参数设置"""
data_root = os.path.abspath(os.path.dirname(__file__)) + '/data/face/'
image_size = (96, 96)           # 输入图片尺寸
batch_size = 600                # 训练时的batch大小
workers = 2                     # dataloader的workers数量
in_channels = 100               # 生成器输入通道数
out_channels = 3                # 生成器输出通道数
lr = 0.0002                     # 学习率
beta1 = 0.5                     # adam优化器的参数
epochs = 2                      # 训练轮数
device = torch.device('cpu')
gpu_num = 0
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_num = torch.cuda.device_count()


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


generator_losses = []
discriminator_losses = []


def train(discriminator, generator, optimizerD, optimizerG):
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            # 训练discriminator
            discriminator.zero_grad()
            real_img = data.to(device)
            real_out = discriminator(real_img).view(-1)
            real_labels = torch.ones(len(real_out)).to(device)
            real_loss = criterion(real_out, real_labels)
            real_loss.backward()

            noise = torch.randn(batch_size, in_channels, 1, 1, device=device)
            fake_img = generator(noise).detach()
            fake_out = discriminator(fake_img).view(-1)
            fake_labels = torch.zeros(len(fake_out)).to(device)
            fake_loss = criterion(fake_out, fake_labels)
            fake_loss.backward()

            discriminator_loss = real_loss + fake_loss

            optimizerD.step()

            # 训练generator
            generator.zero_grad()
            noise = torch.randn(batch_size, in_channels, 1, 1, device=device)
            fake_img = generator(noise)
            fake_out = discriminator(fake_img).view(-1)
            real_labels = torch.ones(len(fake_out)).to(device)
            fake_loss = criterion(fake_out, real_labels)
            fake_loss.backward()

            generator_loss = fake_loss

            optimizerG.step()

            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch, epochs, i, len(dataloader),
                         discriminator_loss.item(), generator_loss.item()))
                discriminator_losses.append(discriminator_loss)
                generator_losses.append(generator_loss)

        test_noise = torch.randn(64, in_channels, 1, 1, device=device)
        with torch.no_grad():
            fake = generator(test_noise).detach().cpu()
        show_data(fake, 'generated images')

        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
        }, './checkpoints/DCGAN/epoch_%d.pth' % epoch)


if __name__ == "__main__":

    batch = next(iter(dataloader))
    show_data(batch[:64], description='training images')

    # 构建生成器
    generator = Generator(in_channels, out_channels).to(device)
    generator.apply(weights_init)
    print(generator)

    # 构建判别器
    discriminator = Discriminator(out_channels).to(device)
    discriminator.apply(weights_init)
    print(discriminator)

    # 是否使用多块GPU
    if (device.type == 'cuda') and (gpu_num > 1):
        print("使用%d块GPU训练!" % gpu_num)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # 损失函数
    criterion = nn.BCELoss()

    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    train(discriminator, generator, optimizerD, optimizerG)

    g_loss = {'loss_list': generator_losses, 'description': 'g_loss'}
    d_loss = {'loss_list': discriminator_losses, 'description': 'd_loss'}
    plot_loss(g_loss, d_loss)
    