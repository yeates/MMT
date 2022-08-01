import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, m=None):
        scores = torch.matmul(query, key.transpose(-2, -1)
                                ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class ADN(nn.Module):
    '''
    Auxiliary DeNormalization
    '''
    def __init__(self, channel, auxiliary_channel):
        super().__init__()
        scale_dic = {256: 32, 128: 64, 64: 128}
        self.norm = nn.LayerNorm([channel, scale_dic[channel], scale_dic[channel]], elementwise_affine=False)
        self.embed = nn.Sequential(
            nn.Conv2d(auxiliary_channel, channel, kernel_size=1, padding=0),
            # nn.ReLU(True),
            )
        self.scale = nn.Sequential(     
            nn.Conv2d(channel, channel, kernel_size=1, padding=0),
            # nn.ReLU(True),
            )
        self.bias = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1, padding=0),
            # nn.ReLU(True),
            )
        
    def forward(self, x, auxiliary):
        '''
        - x: [batch_size, channel, h_prime, w_prime]
        - parsing: [batch_size, categories_num, h, w] 
        '''
        auxiliary_map = self.embed(auxiliary)
        gamma, beta = self.scale(auxiliary_map), self.bias(torch.cat([x, auxiliary_map], dim=1))
        return self.norm(x) * gamma + beta


class MSSA(nn.Module):
    '''
    MSSA with Auxiliary DeNormalization.
    - auxiliary_type: 'feature', 'pred', 'hybrid'
    '''
    def __init__(self, channel, auxiliary_type, feature_channel=0, pred_seg_channel=0, pred_edge_channel=0):
        super().__init__()
        self.auxiliary_type = auxiliary_type # ['feature', 'pred', 'hybrid']
        if auxiliary_type == 'feature':
            pred_seg_channel = pred_edge_channel = 0
        elif auxiliary_type == 'pred':
            feature_channel = 0
        
        self.patchsize = [(2, 2), (4, 4)]
        self.query_embedding = nn.Conv2d(
            channel, channel, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            channel, channel, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            channel, channel, kernel_size=1, padding=0)
        
        self.denorm_1 = ADN(channel, feature_channel*2 + pred_seg_channel + pred_edge_channel)
        self.denorm_2 = ADN(channel, feature_channel*2 + pred_seg_channel + pred_edge_channel)
        self.scale = nn.Sequential(     
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.Tanh())
        self.bias = nn.Sequential(     
            nn.Conv2d(channel, channel, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
            )
        self.attention = Attention()

    def forward(self, x, edge=None, seg=None, pred_edge=None, pred_seg=None):
        input_x = x
        if self.auxiliary_type == 'feature':
            auxiliary = torch.cat([edge, seg], axis=1)
        elif self.auxiliary_type == 'pred':
            auxiliary = torch.cat([pred_edge, pred_seg], axis=1)
        elif self.auxiliary_type == 'hybrid':
            auxiliary = torch.cat([edge, seg, pred_edge, pred_seg], axis=1)
        
        b, c, h, w = x.size()
        d_k = c // len(self.patchsize)
        output = []
        # patch embedding
        _query = self.query_embedding(x)    # [bs,256,h,w]
        _key = self.key_embedding(x)        # [bs,256,h,w]
        _value = self.value_embedding(x)    # [bs,256,h,w]
        # norm
        _query = self.denorm_1(_query, auxiliary)
        _key = self.denorm_1(_key, auxiliary)
        _value = self.denorm_1(_value, auxiliary)
        # multi-head self-attention
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            query = query.view(b, d_k, out_h, height, out_w, width)
            query = query.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                b,  out_h*out_w, d_k*height*width)
            key = key.view(b, d_k, out_h, height, out_w, width)
            key = key.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                b,  out_h*out_w, d_k*height*width)
            value = value.view(b, d_k, out_h, height, out_w, width)
            value = value.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                b,  out_h*out_w, d_k*height*width)
            y, _ = self.attention(query, key, value)
            y = y.view(b, out_h, out_w, d_k, height, width)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        # skip-connection
        mid_x = input_x + output
        # norm
        output = self.denorm_2(mid_x, auxiliary)
        # feedforward
        gamma = self.scale(output)
        beta = self.bias(output)
        res = mid_x * gamma + beta
        return res