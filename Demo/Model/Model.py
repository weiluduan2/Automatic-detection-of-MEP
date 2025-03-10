import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # 加载预训练的DINO模型
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        # 修改第一层卷积以接受单通道输入
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # 将原始3通道权重平均到1通道
            self.conv1.weight.data = self.backbone.conv1.weight.data.mean(dim=1, keepdim=True)
        self.backbone.conv1 = self.conv1

    def forward(self, x1, x2):
        # 输入x1, x2: [batch_size, 1, H, W]
        feat1 = self.backbone(x1)
        feat2 = self.backbone(x2)

        # 特征归一化
        feat1 = F.normalize(feat1, dim=-1, p=2)
        feat2 = F.normalize(feat2, dim=-1, p=2)

        similarity = F.cosine_similarity(feat1, feat2)
        logits = torch.stack([1 - similarity, similarity], dim=1)

        return logits, similarity


"""
使用方式：
model = Network()
# img1, img2 shape: [batch_size, 1, H, W]
pred, sim = model(img1, img2)
"""