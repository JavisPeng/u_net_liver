# u_net_liver

## 数据准备
项目文件分布如下
```
  --project
  	main.py
  	 --data
   		--train
   		--val
```

数据和权重可以使用百度云下载 链接：https://pan.baidu.com/s/10E0CIehRMUA7zepmSbtjjA 提取码：wabv 

## 模型训练
```
python main.py train
```

## 测试模型训练
加载权重，默认保存最后一个权重
```
python main.py test --ckp=weights_19.pth
```

![](https://img-blog.csdn.net/20180508083935908)
