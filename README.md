# PyTorch Hourglass

[CenterNet](https://github.com/xingyizhou/CenterNet) | [CornerNet](https://github.com/princeton-vl/CornerNet) 骨干网络 `Hourglass-104` 的PyTorch实现

## 环境

python == 3.9
pytorch == 1.10

## 使用

调用 `make_connected_hourglass` 获取一个Hourglass网络

参数说明:

`channels`: 网络内部使用的通道数量

`group_res_num`: 一组残差网络包含的残差模块数量

`nesting_times`: 每一个Hourglass网络递归嵌套多少次Hourglass网络自身

`heatmap_dimensions`: 生成的Heatmap维度

`stack_num`: 叠加多少个Hourglass网络

`pool_kernel`: 每一个Hourglass网络头部的`nn.MaxPool2d`层的参数

`pool_stride`: 同上

`enlarger_kernel`: 上采样层放大的倍数

## 导出为ONNX

在`pytorch==1.10`环境下，ONNX模型导出成功。

可以调用`torch.onnx.export(model, input, "myModel.onnx")`将训练好的模型导出为onnx。

在onnxruntime中的使用方法与大部分端到端网络相同。

> 代码内部加入了较为详细的注释，欢迎大家一起学习交流
>
> QQ: 3525904273

## 注意事项

由于Hourglass头部会对输入特征图执行下采样，导致 __长宽减半__。

因此请确保输入图片的 __长宽__ 可以被 __2 ^ nesting_times__ 整除。
