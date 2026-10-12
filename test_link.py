import torch
import torch.nn as nn
# import torchvision.transforms as transforms
from PIL import Image
# 加载预训练的深度估计模型
model = torch.models.resnet18(pretrained=True)
# 去掉最后一层全连接层
model = nn.Sequential(*list(model.children())[:-1])
# 设置模型为评估模式
model.eval()
# 加载输入图像并进行预处理
image_path = 'input_image.jpg'
image = Image.open(image_path)
input_tensor = image
# 使用深度学习模型进行深度估计
with torch.no_grad():
    output_tensor = model(input_tensor)
# 后处理输出结果
depth_map = output_tensor.squeeze().numpy()
# 显示深度图像
depth_image = Image.fromarray((depth_map * 255).astype('uint8'))
depth_image.show()