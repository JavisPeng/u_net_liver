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

数据和权重可以使用百度云下载

## 模型训练
python main.py #在main调用train()方法

## 测试模型训练
加载权重，默认保存最后一个权重
python main.py #在main调用infer()方法
