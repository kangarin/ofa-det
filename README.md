## 记录一下 防止猪脑过载
这里主要做一个ofa的切换demo，就用那种模拟正弦波fps切模型的思路

make_model_example.py 可以获得一个未训练的目标检测模型

search_example.py 可以在本地搜索分类模型的帕累托最优

因为网络问题所以先把once-for-all clone到根目录下面吧，不然hub加载会有问题