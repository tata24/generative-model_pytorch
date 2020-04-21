# gan_pytorch
在卷积网络保持一致的情况下，分别用AutoEncoder,VariationalAutoEncoder和DCGAN来生成动漫人物头像，每个模型均训练50个epoch
## **AutoEncoder**
AE由编码器和解码器两部分组成，编码器将图像压缩至低维度的隐特征，解码器将隐特征重构为图像。AE的目标就是最小化输入与输出之间的重构误差，通常以MSE作为损失函数，其缺点是生成的图像容易出现模糊。由于AE将图像压缩为固定的编码，因此模型的仅能生成与输入类似的输出，模型的生成能力不强。  
AE的生成结果：  
<div align=center><img src="https://github.com/Lijingkan/gan_pytorch/blob/master/images/ae_img.jpeg" width="400" height="400" /></div>

## **VariationalAutoEncoder**
VAE在AE的基础上进行改进，它并不直接学习隐特征的编码，而是学习隐特征的分布，使得模型具有一定泛化能力。VAE的损失函数由两部分组成，一部分是与AE一致的重构误差，另一部分是隐特征的后验分布与01正态分布之间的KL散度。由于从隐特征的分布中采样的操作不可导，因此在实现模型中还使用了一个重参数技巧，使得采样操作不用参与梯度计算。为了保证训练有效，kl散度损失前还乘了一个系数来控制两部分损失的比例。
  VAE的生成结果：  
  <div align=center><img src="https://github.com/Lijingkan/gan_pytorch/blob/master/images/vae_img.jpeg" width="400" height="400" /></div>
  

## **DCGAN**
DCGAN由生成器和判别器组成，生成器用来生成图像，判别器用来判断的图像的真假，因此GAN并不指定隐特征的具体分布，而是通过生成器与判别器的对抗来直接学习图像的分布。  
DCGAN的生成结果：  
<div align=center><img src="https://github.com/Lijingkan/gan_pytorch/blob/master/images/dcgan_img.jpeg" width="400" height="400" /></div>

## 数据集链接
```
https://zhuanlan.zhihu.com/p/24767059
```


