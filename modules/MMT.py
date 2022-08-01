import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.MSSA import MSSA
# from utils.functions import AOTBlock, ResnetBlock # for ablation


class GatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, is_leaky=True, bias=False, post=False):
        super().__init__()
        self.post_processing = post
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, bias=bias)
        self.gate = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, bias=bias)
        self.activation = nn.LeakyReLU(0.2, True) if is_leaky else nn.ReLU(True)
        if post:
            self.post = nn.Sequential(nn.BatchNorm2d(out_channel), nn.ReLU(inplace = True))

    def forward(self, x, m=None):
        feat = self.activation(self.conv(x))
        gating = torch.sigmoid(self.gate(x))
        output = feat * gating
        if self.post_processing:
            output = self.post(output)
        return output, gating


class EdgeProjector(nn.Module):
    '''
    output inpainted edge.
    '''
    def __init__(self, in_channel, out_channel=1):
        super(EdgeProjector, self).__init__()
        self.TConv = nn.Sequential(nn.ConvTranspose2d(in_channel, 64, 4, 2, 1),
                                    nn.ReLU(True))
        self.gatedConv = GatedConv2d(67, 32, 3, 1, 1, bias=True)
        self.out = nn.Conv2d(32, out_channel, 3, 1, 1)
    
    def forward(self, x, in_image):
        x = self.TConv(x)
        x, _ = self.gatedConv(torch.cat([x, in_image], axis=1))
        x = torch.sigmoid(self.out(x))
        return x


class SegProjector(nn.Module):
    '''
    output inpainted segmentation.
    '''
    def __init__(self, in_channel, out_channel):
        super(SegProjector, self).__init__()
        self.TConv = nn.Sequential(nn.ConvTranspose2d(in_channel, 64, 4, 2, 1),
                                    nn.ReLU(True))
        self.gatedConv = GatedConv2d(67, 32, 3, 1, 1, bias=True)
        self.out = nn.Conv2d(32, out_channel, 3, 1, 1)
    
    def forward(self, x, in_image):
        x = self.TConv(x)
        x, _ = self.gatedConv(torch.cat([x, in_image], axis=1))
        x = self.out(x)
        return x

    
class RGBTail(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(RGBTail, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class ACBlock(nn.Module):
    def __init__(self, in_channel, out_channel, rates=[1,3,6,9]):
        super(ACBlock, self).__init__()
        self.rates = rates
        self.out_channel = out_channel
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(in_channel, out_channel, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channel+out_channel*2, out_channel, 3, padding=1, dilation=1),
            nn.AdaptiveAvgPool2d((1,1)))    # global average pooling
        self.W = nn.Parameter(torch.Tensor(out_channel, len(rates))) # learnable matrix
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.gate = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1))
        
        self.post = nn.Sequential(nn.BatchNorm2d(out_channel),
                                nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, g_):
        '''
        - x: feature map
        - g_: previous gating
        '''
        g = self.gate(x)    # current gating
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        
        att = torch.flatten(self.fuse(torch.cat([x, g, g_], dim=1)), 1)
        assert self.out_channel == att.shape[-1]
        weight = torch.softmax(torch.tanh(torch.matmul(att, self.W)), dim=1)   # -> [bs * 4]
        
        out = torch.stack(out, dim=1)
        weight = weight[..., None, None, None]
        weighted_out = torch.sum(out * weight, 1)
        
        mask = torch.sigmoid(g)
        results = x * (1 - mask) + weighted_out * mask 
        return self.post(results), mask


class AuxiliaryDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pred_auxiliary_channel, is_tconv=False, is_pred=False):
        super(AuxiliaryDecoder, self).__init__()
        if is_tconv:
            self.conv_block = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding),
                nn.LeakyReLU(0.2, inplace = True)
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                nn.LeakyReLU(0.2, inplace = True)
            )
        if is_pred:
            self.auxiliary_head = nn.Conv2d(out_channel, pred_auxiliary_channel, 3, 1, 1)
    
    def forward(self, x):
        pred = None
        x = self.conv_block(x)
        if hasattr(self, "auxiliary_head"):
            pred = self.auxiliary_head(x)
        return x, pred  # feature, auxiliary structure 


class MMTModule(nn.Module):
    def __init__(self, n_cate_anno, auxiliary_type, layer_size=8, in_channel = 64):
        super(MMTModule, self).__init__()
        self.layer_size = layer_size
        
        # enc_1
        block = [nn.Conv2d(64, 128, 3, 2, 1, bias = False),  nn.BatchNorm2d(128), nn.ReLU(inplace = True)]
        self.enc_1 = nn.Sequential(*block)
        # enc_2
        block = [nn.Conv2d(128, 256, 3, 2, 1, bias = False),  nn.BatchNorm2d(256), nn.ReLU(inplace = True)]
        self.enc_2 = nn.Sequential(*block)
        # enc_3
        self.enc_3 = GatedConv2d(256, 256, 3, 1, 1, bias = False, post=True)
        
        # bottleneck: Adaptive Contextual Bottleneck
        for i in range(0, self.layer_size):
            name = 'bottleneck_{:d}'.format(i+1)
            setattr(self, name, ACBlock(256, 256, [1,2,3,4]))
            
        # fusion: Multi-Scale Spatial-aware Attention
        self.att_3 = MSSA(256, auxiliary_type, 128, n_cate_anno, 1)         # 'hybrid', 128, n_cate_anno, 1
        self.att_2 = MSSA(128, auxiliary_type, 128, n_cate_anno, 1)         # 'hybrid', 128, n_cate_anno, 1
        self.att_1 = MSSA(64, auxiliary_type, 64, n_cate_anno, 1)           # 'hybrid', 64, n_cate_anno, 1
        # dec_3
        block = [nn.Conv2d(512, 256, 3, 1, 1, bias = False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace = True)]
        self.dec_3 = nn.Sequential(*block)
        # dec_2
        block = [nn.ConvTranspose2d(512, 128, 4, 2, 1, bias = False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace = True)]
        self.dec_2 = nn.Sequential(*block)
        # dec_1
        block = [nn.ConvTranspose2d(256, 64, 4, 2, 1, bias = False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace = True)]
        self.dec_1 = nn.Sequential(*block)
        
        is_pred = (auxiliary_type in ["pred", "hybrid"])    # if predict multi-scale structure
        # =====for seg decoder=====
        self.seg_dec_3 = AuxiliaryDecoder(256 + 256, 128, 3, 1, 1, n_cate_anno, is_pred=is_pred)
        self.seg_dec_2 = AuxiliaryDecoder(256 + 256 + 128, 128, 4, 2, 1, n_cate_anno, is_tconv=True, is_pred=is_pred)
        self.seg_dec_1 = AuxiliaryDecoder(128 + 128 + 128, 64, 4, 2, 1, n_cate_anno, is_tconv=True, is_pred=is_pred)
        # =========================
        
        # =====for edge decoder=====
        self.edge_dec_3 = AuxiliaryDecoder(256 + 256, 128, 3, 1, 1, pred_auxiliary_channel=1, is_pred=is_pred)
        self.edge_dec_2 = AuxiliaryDecoder(256 + 256 + 128, 128, 4, 2, 1, pred_auxiliary_channel=1, is_tconv=True, is_pred=is_pred)
        self.edge_dec_1 = AuxiliaryDecoder(128 + 128 + 128, 64, 4, 2, 1, pred_auxiliary_channel=1, is_tconv=True, is_pred=is_pred)
        # =========================
        
    def forward(self, input):
        h_dict = {}  # for the output of enc_N
        h_dict['h_0']= input
        
        # ======= enc ========
        h_key_prev = 'h_0'
        for i in range(1, 4):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            if i != 3:
                h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
            else:
                h_dict[h_key], previous_g = getattr(self, l_key)(h_dict[h_key_prev])
            h_key_prev = h_key
        
        # ===== bottleneck =====
        h = h_dict[h_key]
        for i in range(0, self.layer_size):
            l_key = 'bottleneck_{:d}'.format(i+1)
            h, previous_g = getattr(self, l_key)(h, previous_g)
        # ======================
        
        # ======== dec =========
        ms_pred_anno = []    # multi-scale predicted segmentation
        ms_pred_edge = []    # multi-scale predicted edge
        seg_h = h
        edge_h = h
        for i in range(3, 0, -1):
            enc_h_key = 'h_{:d}'.format(i)
            dec_l_key = 'dec_{:d}'.format(i)
            seg_dec_l_key = 'seg_dec_{:d}'.format(i)
            edge_dec_l_key = 'edge_dec_{:d}'.format(i)
            # (1) seg decoder
            seg_h = torch.cat([seg_h, h_dict[enc_h_key]], dim=1) if i == 3 else torch.cat([seg_h, h, h_dict[enc_h_key]], dim=1)
            seg_h, seg_pred = getattr(self, seg_dec_l_key)(seg_h)
            # (2) edge decoder
            edge_h = torch.cat([edge_h, h_dict[enc_h_key]], dim=1) if i == 3 else torch.cat([edge_h, h, h_dict[enc_h_key]], dim=1)
            edge_h, edge_pred = getattr(self, edge_dec_l_key)(edge_h)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h = getattr(self, dec_l_key)(h)
            # (3) msa
            h = getattr(self, f"att_{i}")(h, edge_h, seg_h, edge_pred, seg_pred)
            ms_pred_anno.append(seg_pred)
            ms_pred_edge.append(edge_pred)
        
        return h, edge_h, seg_h, ms_pred_edge, ms_pred_anno


class MMTNet(nn.Module):
    def __init__(self, n_cate_anno, auxiliary_type):
        super(MMTNet, self).__init__()
        self.conv1 = GatedConv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = GatedConv2d(64, 64, 7, 1, 3, bias=False)
        self.bn20 = nn.BatchNorm2d(64)
        self.conv21 = GatedConv2d(64, 64, 7, 1, 3)
        self.conv22 = GatedConv2d(64, 64, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.MMTModule = MMTModule(n_cate_anno, auxiliary_type)
        self.Tconv = nn.ConvTranspose2d(64, 64, 4, 2, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tail1 = GatedConv2d(67, 32, 3, 1, 1)
        self.tail2 = RGBTail(32,8)
        self.out = nn.Sequential(nn.Conv2d(64,3,3,1,1, bias = False),
                                    nn.Tanh())
        
        self.seg_projector = SegProjector(64, n_cate_anno)
        self.edge_projector = EdgeProjector(64)

    def forward(self, in_image, mask):
        x1, m1 = self.conv1(in_image, mask)
        x1 = F.relu(self.bn1(x1), inplace = True)
        x1, m1 = self.conv2(x1, m1)
        x1 = F.relu(self.bn20(x1), inplace = True)
        x2 = x1
        x2, m2 = x1, m1
        n, c, h, w = x2.size()
        feature_group = [x2.view(n, c, 1, h, w)]
        mask_group = [m2.view(n, c, 1, h, w)]
        
        x2, m2 = self.conv21(x2, m2)
        x2, m2 = self.conv22(x2, m2)
        x2 = F.leaky_relu(self.bn2(x2), inplace = True)
        x2, edge_h, seg_h, ms_pred_edge, ms_pred_anno = self.MMTModule(x2)
        x2 = x2 * m2
        feature_group.append(x2.view(n, c, 1, h, w))
        mask_group.append(m2.view(n, c, 1, h, w))
            
        pred_anno = self.seg_projector(seg_h, in_image)
        pred_edge = self.edge_projector(edge_h, in_image)
        x3 = torch.cat(feature_group, dim = 2)
        m3 = torch.cat(mask_group, dim = 2)
        amp_vec = m3.mean(dim = 2)
        x3 = (x3*m3).mean(dim = 2) /(amp_vec+1e-7)
        x3 = x3.view(n, c, h, w)
        m3 = m3[:,:,-1,:,:]
        x4 = self.Tconv(x3)
        x4 = F.leaky_relu(self.bn3(x4), inplace = True)
        m4 = F.interpolate(m3, scale_factor = 2)
        x5 = torch.cat([in_image, x4], dim = 1)
        m5 = torch.cat([mask, m4], dim = 1)
        x5, _ = self.tail1(x5, m5)
        x5 = F.leaky_relu(x5, inplace = True) 
        x6 = self.tail2(x5)
        x6 = torch.cat([x5,x6], dim = 1)
        output = self.out(x6)
        output = (output + 1) / 2.
        return output, [pred_edge, pred_anno, (ms_pred_edge, ms_pred_anno)]
    