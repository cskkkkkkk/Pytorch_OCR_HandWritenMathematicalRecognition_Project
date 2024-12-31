import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        #params表示一些超参数
        self.params = params
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']
        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

    #cnn_features是CNN提取的特征，cnn_features_trans是CNN提取的特征的转置，hidden是decoder的隐藏状态，alpha_sum是上一步的attention权重，image_mask是图像的mask
    def forward(self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        # permute(0, 2, 3, 1) 后，张量的维度被重新排列为 [𝐵,𝐻,𝑊,𝐶][B,H,W,C]
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0,2,3,1))
        #query表示[B,1,1,attention_dim]，alpha_score表示注意力权重
        alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0,2,3,1))
        #energy是注意力机制中的打分向量，衡量不同位置的特征对解码器当前时刻的输出有多重要
        energy = self.alpha_convert(alpha_score)
        #放置exp数值溢出了
        energy = energy - energy.max()
        #squeeze会把为1的维度去掉
        energy_exp = torch.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        #alpha_sum: 累积注意力分布，用于覆盖机制，防止关注区域过度集中。
        alpha_sum = alpha[:,None,:,:] + alpha_sum
        #cnn_features是CNN提取的特征图形状为：[B,C,H,W],sum(-1)表示对最后的两个维度进行求和，最后得到[B,C]
        #得到一个聚合的上下文向量 context_vector
        context_vector = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)
        return context_vector, alpha, alpha_sum
