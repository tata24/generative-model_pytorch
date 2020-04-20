# gan_pytorch
在卷积网络保持一致，分别用AE,VAE,DCGAN来生成动漫人物头像，每个模型均训练50个epoch
## **AutoEncoder**
AE由编码器和解码器两部分组成，编码器将图像压缩至低维度的隐特征，解码器将隐特征重构为图像。AE的目标就是最小化输入与输出之间的重构误差，通常以MSE作为损失函数，其缺点是生成的图像容易出现模糊。由于AE将图像压缩为固定的编码，因此模型的仅能生成与输入类似的输出，模型的生成能力不强。  
AE的实验结果：  
<div align=center><img src="https://github.com/Lijingkan/gan_pytorch/blob/master/images/ae_img.png" width="400" height="400" /></div>

## **VAE**
## **DCGAN**
## 数据集链接
```
https://zhuanlan.zhihu.com/p/24767059
```


