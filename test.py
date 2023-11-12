import torch

import torch.nn.functional as F

# 你的输入数据，5行2列的二维数组
input_data = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0],
                          [0.5, 0.2],
                          [2.5, 1.5],
                          [4.0, 3.0]])

# 使用torch.nn.functional.softmax函数对每一行进行softmax操作
softmax_output = F.softmax(input_data, dim=1)

print(softmax_output)