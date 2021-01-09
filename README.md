# 人脸表情识别

https://github.com/lianghaofa/computer-vision

## 简介
使用卷积神经网络构建整个系统，在尝试了Gabor、LBP等传统人脸特征提取方式基础上，深度模型效果显著。在FER2013、JAFFE和CK+三个表情识别数据集上进行模型评估。


## 环境部署
基于Python3和Keras2（TensorFlow后端），具体依赖安装如下(推荐使用conda虚拟环境)。
```shell script
git clone https://github.com/lianghaofa/computer-vision
cd FacialExpressionRecognition
conda create -n FER python=3.6
source activate FER
conda install cudatoolkit=10.1
conda install cudnn=7.6.5
pip install -r requirements.txt
```
如果你是Linux用户，直接执行根目录下的`env.sh`即可一键配置环境，执行命令为`bash env.sh`。


## 网络设计
使用经典的卷积神经网络，模型的网络结构设计如下，输入层后加入(1,1)卷积层增加非线性表示且模型层次较浅，参数大量参数集中在全连接层。
<div align="center"><img src="./assets/CNN.png" /></div>
<div align="center"><img src="./assets/model.png" /></div>


## 模型训练
主要在FER2013、JAFFE、CK+上进行训练，JAFFE给出的是半身图因此做了人脸检测。最后在FER2013上Pub Test和Pri Test均达到65%左右准确率（该数据集爬虫采集存在标签错误、水印、动画图片等问题），JAFFE和CK+5折交叉验证均达到97%左右准确率（这两个数据集为实验室采集，较为准确标准）。

执行下面的命令将在指定的数据集（fer2013或jaffe或ck+）上按照指定的batch_size训练指定的轮次。训练会生成对应的可视化训练过程，下图为在三个数据集上训练过程的共同绘图。

```shell
python src/train.py --dataset fer2013 --epochs 300 --batch_size 32
```


## 模型应用
与传统方法相比，卷积神经网络表现更好，使用该模型构建识别系统。预测时对一张图片进行水平翻转、偏转15度、平移等增广得到多个概率分布，将这些概率分布加权求和得到最后的概率分布，此时概率最大的作为标签（也就是使用了推理数据增强）。

## 总结
静态/动态都有一定的应用空间，后者更接近真实客观表情理解。可以看出，除了后面网络集成，网络级联更多的用在刷分，工程意义不大。在对模型进行训练前，需要对数据进行预处理，比如人脸检测，人脸对齐，亮度，对比度归一化，姿态归一化，保证输入数据干扰影响较少。数据增强与常见的图片数据扩展一致，深度网络包括CNN分类/识别/RNN分类，与常见分类任务一致。可以用预训练的分类/人脸识别网络fineturn，可以针对表情修改对应的损失函数，也可以局部或者针对某些层，块进行改进。也可以将DL和Handcrafted特征结合，增加网络输入的多样性。


