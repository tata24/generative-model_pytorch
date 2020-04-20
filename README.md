# gan_pytorch
在卷积网络保持一致的情况下，分别用AE,VAE,DCGAN来生成动漫人物头像，每个模型均训练50个epoch
## **AutoEncoder**
AE由编码器和解码器两部分组成，编码器将图像压缩至低维度的隐特征，解码器将隐特征重构为图像。AE的目标就是最小化输入与输出之间的重构误差，通常以MSE作为损失函数，其缺点是生成的图像容易出现模糊。由于AE将图像压缩为固定的编码，因此模型的仅能生成与输入类似的输出，模型的生成能力不强。  
AE的实验结果：  
<div align=center><img src="https://github.com/Lijingkan/gan_pytorch/blob/master/images/ae_img.png" width="400" height="400" /></div>

## **VAE**
## **DCGAN**
DCGAN由生成器和判别器组成，生成器用来生成图像，判别器用来判断的图像的真假，因此GAN并不指定隐特征的具体分布，而是通过生成器与判别器的对抗来直接学习图像的分布。  
DCGAN的实验结果：  
<div align=center><img src="https://github.com/Lijingkan/gan_pytorch/blob/master/images/dcgan_img.png" width="400" height="400" /></div>

## 数据集链接
```
https://zhuanlan.zhihu.com/p/24767059
```


