# ImageSegmentation

## Background

针对 “2024人工智能大赛” 钢材表面缺陷检测与分割问题提供的解题思路。 

## File Structure

`data`：存放训练集和测试集以及对应的标签

`models`：训练得到的模型，存储在 `.pth` 文件中

`preprocessing`：对图片的预处理，将图片转化为三维张量，存储在 `.pt` 文件中

`tensors`：存储 `.pt` 文件

`test`：对模型进行测试

`training`：对模型进行训练

`generate`：生成结果文件，前三个文件夹

## How to run the project

1. 逐个运行 `./preprocessing` 文件夹中的文件

2. 逐个运行 `./training` 文件夹中的文件

3. 逐个运行 `./test` 文件夹中的文件

## Evaluation Metrics

- Class1 IoU、Class2 IoU、Class3 IoU、mIoU 、FPS 、模型参数量

