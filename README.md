# unet liver
Unet network for liver CT image segmentation
## data preparation
structure of project
```
  --project
  	main.py
  	 --data
   		--train
   		--val
```
data and trained weight link: https://pan.baidu.com/s/1dgGnsfoSmL1lbOUwyItp6w code: 17yr 


## training
```
python main.py train
```

## testing
load the last saved weight
```
python main.py test --ckpt=weights_19.pth
```
----

## 数据准备
项目文件分布如下
```
  --project
  	main.py
  	 --data
   		--train
   		--val
```

数据和权重可以使用百度云下载 链接: 

链接: https://pan.baidu.com/s/1dgGnsfoSmL1lbOUwyItp6w 提取码: 17yr

## 模型训练
```
python main.py train
```

## 测试模型训练
加载权重，默认保存最后一个权重
```
python main.py test --ckpt=weights_19.pth
```
## 多类别
修改2个地方即可：unet最后一层的通道数设置为类别数；损失函数使用CrossEntropyLoss
```python
bath_size,img_size,num_classes=2,3,4
#model = Unet(3, num_classes)
criterion = nn.CrossEntropyLoss()
#assume the pred is the output of the model
pred=torch.rand(bath_size,num_classes,img_size,img_size)
target=torch.randint(num_classes,(bath_size,img_size,img_size))
loss=criterion(pred,target)
```

## Demo
![liver](https://img-blog.csdn.net/20180508083935908)
