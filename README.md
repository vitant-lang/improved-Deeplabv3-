# improved-Deeplabv3-
改进后的 Deeplabv3+ 分别测试了CBAM ECA SE 模块。


# 网络改进部分如图所示 

![image](https://github.com/vitant-lang/improved-Deeplabv3-/assets/75409802/16c1296e-d669-4c75-91f7-601bc02965d3)


显存暴毙问题 如果一开始无法训练，应该是batch size的问题，如果训练一会爆了，个人经验是调整numworker的值，（在文件里CTRL加F搜）。


