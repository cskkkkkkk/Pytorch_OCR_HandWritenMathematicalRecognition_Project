import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.densenet import DenseNet
from models.counting import CountingDecoder as counting_decoder
from counting_utils import gen_counting_label


class CAN(nn.Module):
    def __init__(self, params=None):
        super(CAN, self).__init__()
        self.params = params
        #是否使用标签掩码
        #可变长度序列: 例如自然语言处理（NLP）中，输入序列长度不同，为了保持张量的统一形状，使用填充（padding）来补全短序列，这些填充值需要被掩盖（mask）。
        #部分标签无效: 数据中有些标签可能是缺失值或者是辅助生成的，不应参与模型训练。
        #加权计算: 在训练过程中，如果某些标签的重要性较低，可以通过掩码降低其权重。
        self.use_label_mask = params['use_label_mask']
        #加载一个densenet用于图像特征提取
        self.encoder = DenseNet(params=self.params)
        self.in_channel = params['counting_decoder']['in_channel']
        self.out_channel = params['counting_decoder']['out_channel']
        #分别使用不同大小卷积核（3和5）从图像特征提取的结果中提取counting特征
        self.counting_decoder1 = counting_decoder(self.in_channel, self.out_channel, 3)
        self.counting_decoder2 = counting_decoder(self.in_channel, self.out_channel, 5)
        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)
        #如果 use_label_mask=True，会在损失计算中考虑标签掩码
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        #平滑的 L1 损失，用于计数任务中预测值和目标值之间的误差计算。
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

    def forward(self, images, images_mask, labels, labels_mask, is_train=True):
        cnn_features = self.encoder(images)
        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        counting_labels = gen_counting_label(labels, self.out_channel, True)

        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        counting_preds = (counting_preds1 + counting_preds2) / 2
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)

        word_probs, word_alphas = self.decoder(cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=is_train)
        word_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels.view(-1))
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss
        return word_probs, counting_preds, word_average_loss, counting_loss
